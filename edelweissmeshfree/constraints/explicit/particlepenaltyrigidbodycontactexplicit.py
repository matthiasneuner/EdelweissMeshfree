from collections.abc import Callable, Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import NextIncrement
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmexplicitconstraintbase import (
    MPMExplicitConstraintBase,
)
from edelweissmeshfree.constraints.base.pointsofparticles import (
    SpringsOnPointsOfParticle,
    addBoundOfAbsoluteStiffnessOfPointsOfParticle,
    addForcesOnPointsOfParticle,
    locateTheNodesOfParticles,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyContactImplicitSurfaceConstraintExplicit(MPMExplicitConstraintBase):
    """
    Penalty contact constraint for arbitrary implicit surfaces in EXPLICIT simulations.

    Parameters
    ----------
    particle
        The particle to which the constraint is applied.
    implicit_function
        Returns signed distance. Convention: Negative = Inside (Penetration).
    gradient_function
        Returns gradient vector. Convention: Points OUTWARD (Normal).
    model
        The MPM model instance.
    location
        Where to apply the constraint on the particle: "center", "face", or "vertex".
    faceIDs
        List of face IDs to apply constraint if location is "face".
    vertexIDs
        List of vertex IDs to apply constraint if location is "vertex".
    penaltyParameter
        The penalty stiffness parameter. Higher values reduce penetration but can cause numerical issues.
    doProximityCheck
        If True, constraints are only activated when particles are within a certain proximity to the surface.
    proximityFactor
        Multiplier for the proximity distance threshold, based on particle size. Default is 2.
    interpolationLinearizedAboutCenter
        False (default): measure the field at the point and distribute the force there with the shape functions at
        the point, N. True: with the particle's interpolation linearized about its center, G, with which the particle
        moves its own points (symmetric and conservative). See
        :meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`.
    """

    #: Carries nothing from one increment to the next.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        implicit_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        model: MPMModel,
        location: str = "center",
        faceIDs: list[int] = None,
        vertexIDs: list[int] = None,
        penaltyParameter: float = 1e5,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
        interpolationLinearizedAboutCenter: bool = False,
    ):
        self._name = name
        self.interpolationLinearizedAboutCenter = interpolationLinearizedAboutCenter
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        # the nodes of the particle's kernel functions, and where they sit in the block (see updateConnectivity)
        self._nodes = []
        self._localNodes = np.zeros(0, dtype=int)

        self._implicit_function = implicit_function
        self._gradient_function = gradient_function

        self.reactionForce = 0.0

        def _getConstraintLocations():
            if location == "center":
                return [particle.getCenterCoordinates()]
            elif location == "face":
                if faceIDs is None:
                    raise ValueError("faceID must be specified when location is 'face'.")
                ids = [faceIDs] if isinstance(faceIDs, int) else faceIDs
                return [particle.getFaceCoordinates(idx) for idx in ids]
            elif location == "vertex":
                if vertexIDs is None:
                    raise ValueError("vertexID must be specified when location is 'vertex'.")
                ids = [vertexIDs] if isinstance(vertexIDs, int) else vertexIDs
                vertices = particle.getVertexCoordinates()
                return [vertices[idx] for idx in ids]
            return []

        self._getConstraintLocations = _getConstraintLocations
        self._domainSize = model.domainSize
        self._constrained_points = self._getConstraintLocations()
        self._particle = particle

        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck

        if self._doProximityCheck:
            self._proximityDist = particle.getVolumeUndeformed() ** (1.0 / self._domainSize) * proximityFactor

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return [[self._field]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list:
        return []

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model: MPMModel) -> bool:
        """
        Updates the constraint connectivity by refreshing active nodes and positions.

        Parameters
        ----------
        model
            The MPM model instance, used to access grid information if needed.

        Returns
        -------
        bool
            True if the connectivity has changed (e.g., active nodes or positions), False otherwise.
        """
        self._constrained_points = self._getConstraintLocations()

        # the nodes of the particle's kernel functions, and where they sit in the constraint's block
        localIndexOfNode = {}
        [self._localNodes] = locateTheNodesOfParticles([self._particle.kernelFunctions], localIndexOfNode)
        nodes = list(localIndexOfNode)
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        # within reach if one of the constrained points is close enough to the surface (signed distance below it)
        self.withinReach = not self._doProximityCheck or any(
            self._implicit_function(p_coord) < self._proximityDist for p_coord in self._constrained_points
        )

        if not self.active:
            self.reactionForce = 0.0

        return hasChanged

    def _springs(self, timeStep: TimeStep, nextIncrement: NextIncrement | None = None) -> SpringsOnPointsOfParticle:
        """The penalty springs of the contact at the particle's current position: one of the penalty stiffness at every
        constrained point inside the surface (negative signed distance :math:`d`), along the outward unit normal (the
        normalized gradient of the signed distance; none where the gradient vanishes), pushing the point out with
        :math:`f = k\\, |d|`. The force (:meth:`computeKernelsExplicit`) and the bound of the stiffness
        (:meth:`addBoundOfAbsoluteStiffnessTimes`) go through these springs; the bound over the next increment with the
        smallest signed distance of each point within it
        (:meth:`~edelweissfe.constraints.base.constraintbase.NextIncrement.smallestGaps`).

        Parameters
        ----------
        timeStep
            The current time step.
        nextIncrement
            The next increment, for the bound over it; None for the state now.

        Returns
        -------
        SpringsOnPointsOfParticle
            The springs; none if no point is inside.
        """

        distances = np.array([self._implicit_function(p) for p in self._constrained_points])
        if nextIncrement is not None:

            def distancesAt(U, dU, timeStep):
                moved = self.pointsOfParticleMovedBy(
                    dU, self._particle, self._constrained_points, self._localNodes, self._fieldSize
                )
                return np.array([self._implicit_function(p) for p in moved])

            zero = np.zeros(self.nDof)
            distances = nextIncrement.smallestGaps(distancesAt, zero, zero, timeStep)
        inside = [p for p, distance in zip(self._constrained_points, distances) if distance < 0]
        directions = np.zeros((len(inside), self._domainSize))
        for row, p in enumerate(inside):
            gradient = self._gradient_function(p)
            norm = np.linalg.norm(gradient)
            if norm > 1e-14:
                directions[row] = gradient / norm
        return SpringsOnPointsOfParticle(
            # the vector the force is distributed with (see interpolationVectorAtPointOf)
            np.array([self.interpolationVectorAtPointOf(self._particle, p).flatten() for p in inside]).reshape(
                (len(inside), len(self._localNodes))
            ),
            directions,
            -self._penaltyParameter * distances[distances < 0],
            np.full(len(inside), self._penaltyParameter),
        )

    def computeKernelsExplicit(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, timeStep: TimeStep):
        """Add the residual contribution of the penalty forces: the negative of the forces with which the surface
        pushes the particle back, evaluated at the particle's current position (:meth:`_springs`).

        Parameters
        ----------
        U_np
            The current solution; unused, the penetration follows from the particle's current position.
        dU
            The solution increment; likewise unused.
        P
            The residual contribution to augment.
        timeStep
            The current time step; unused.
        """

        springs = self._springs(timeStep)
        addForcesOnPointsOfParticle(
            P.reshape(-1, self._fieldSize),
            self._localNodes,
            springs.interpolations,
            springs.forces[:, None] * springs.directions,
        )
        self.reactionForce = springs.forces.sum()

    def addBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
        nextIncrement: NextIncrement | None = None,
    ):
        """Add the row sums of the absolute value of the stiffness of the penalty springs (:meth:`_springs`), weighted
        by column (see
        :func:`~edelweissmeshfree.constraints.base.pointsofparticles.addBoundOfAbsoluteStiffnessOfPointsOfParticle`),
        over the next increment if it is given: a point that could close within it has its spring.

        Parameters
        ----------
        U_np
            The current solution; unused.
        dU
            The solution increment; unused.
        weights
            The weight of each column.
        bound
            The row sums to augment.
        timeStep
            The current time step.
        nextIncrement
            The next increment the solver may take; None to bound the state now alone.
        """

        addBoundOfAbsoluteStiffnessOfPointsOfParticle(
            bound, weights, self._localNodes, self._springs(timeStep, nextIncrement)
        )


def ParticlePenaltyContactImplicitSurfaceConstraintExplicitFactory(
    baseName: str,
    implicit_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
    interpolationLinearizedAboutCenter: bool = False,
):
    """
    Factory function to create Explicit Penalty Contact constraints for collections.

    Parameters
    ----------
    baseName
        Base name for constraints. Unique IDs will be appended.
    implicit_function
        Function that returns signed distance. Convention: Negative = Inside (Penetration).
    gradient_function
        Function that returns gradient vector. Convention: Points OUTWARD (Normal).
    particleCollection
        Either a list of particles or an EntityBasedSurface. If EntityBasedSurface, constraints are created for each face.
    model
        The MPM model instance.
    location
        Where to apply the constraint on the particle: "center", "face", or "vertex
    faceIDs
        List of face IDs to apply constraint if location is "face". Ignored if particleCollection
        is an EntityBasedSurface (constraints will be created for all faces).
    vertexIDs
        List of vertex IDs to apply constraint if location is "vertex".
    penaltyParameter
        The penalty stiffness parameter. Higher values reduce penetration but can cause numerical issues.
    doProximityCheck
        If True, constraints are only activated when particles are within a certain proximity to the surface.
    proximityFactor
        Multiplier for the proximity distance threshold, based on particle size. Default is 2.
    interpolationLinearizedAboutCenter
        Passed to every constraint: False (default) the shape functions at the point, N; True the particle's
        interpolation linearized about its center, G.
    """
    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraints[name] = ParticlePenaltyContactImplicitSurfaceConstraintExplicit(
                    name,
                    p,
                    implicit_function,
                    gradient_function,
                    model,
                    location="face",
                    faceIDs=faceID,
                    penaltyParameter=penaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                    interpolationLinearizedAboutCenter=interpolationLinearizedAboutCenter,
                )
        return constraints

    elif isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraints[name] = ParticlePenaltyContactImplicitSurfaceConstraintExplicit(
                name,
                p,
                implicit_function,
                gradient_function,
                model,
                location,
                faceIDs,
                vertexIDs,
                penaltyParameter,
                doProximityCheck,
                proximityFactor,
                interpolationLinearizedAboutCenter=interpolationLinearizedAboutCenter,
            )
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")
