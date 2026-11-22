from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
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


class ParticlePenaltyCartesianBoundaryConstraintExplicit(MPMExplicitConstraintBase):
    r"""
    This class implements a Penalty contact boundary constraint for EXPLICIT simulations.
    It prevents a particle from penetrating a defined boundary surface by applying
    penalty forces directly to the grid nodes.

    The constraint is applied based on the current position of the particle relative to the boundary.
    If penetration is detected, the penalty force is

    .. math::

        \boldsymbol{F}_{\mathrm{penalty}} = -k\, g\, \boldsymbol{n},

    where :math:`g` is the penetration depth, :math:`k` the penalty parameter and :math:`\boldsymbol{n}` the normal.

    Parameters
    ----------
    name : str
        Name of the constraint.
    particle : BaseParticle
        The particle to which the constraint is applied.
    component : int
        The component (0 for x, 1 for y, 2 for z) of the displacement field.
    boundaryPosition : float
        The position of the boundary along the specified component.
    model : MPMModel
        The simulation model.
    location : str
        The location on the particle to apply the constraint ("center", "face", "vertex").
    faceIDs : list[int]
        The face IDs if location is "face".
    vertexIDs : list[int]
        The vertex IDs if location is "vertex".
    velocity : float
        The velocity of the boundary along the constrained component.
    penaltyParameter : float
        The penalty stiffness.
    doProximityCheck : bool
        Whether to perform a proximity check to deactivate distant constraints.
    proximityFactor : float
        The factor for proximity checking distance.
    """

    #: Carries nothing from one increment to the next.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        component: int,
        boundaryPosition: float,
        model,
        location: str = "center",
        faceIDs: list[int] = None,
        vertexIDs: list[int] = None,
        velocity: float = 0.0,
        penaltyParameter: float = 1e5,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
    ):
        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)

        # the nodes of the particle's kernel functions, and where they sit in the block (see updateConnectivity)
        self._nodes = []
        self._localNodes = np.zeros(0, dtype=int)

        self._component = component
        self.reactionForce = 0.0

        self._boundaryPosition = boundaryPosition
        self._vertexIDs = vertexIDs
        self._faceIDs = faceIDs

        def _getConstraintLocations():
            if location == "center":
                return [particle.getCenterCoordinates()]
            elif location == "face":
                if self._faceIDs is None:
                    raise ValueError("faceID must be specified when location is 'face'.")
                ids = [self._faceIDs] if isinstance(self._faceIDs, int) else self._faceIDs
                return [particle.getFaceCoordinates(idx) for idx in ids]
            elif location == "vertex":
                if self._vertexIDs is None:
                    raise ValueError("vertexID must be specified when location is 'vertex'.")
                ids = [self._vertexIDs] if isinstance(self._vertexIDs, int) else self._vertexIDs
                vertices = particle.getVertexCoordinates()
                return [vertices[idx] for idx in ids]
            return []

        self._getConstraintLocations = _getConstraintLocations
        domainSize = model.domainSize
        self._constrained_points = self._getConstraintLocations()
        self._particle = particle

        if component < 0 or component >= domainSize:
            raise ValueError(f"Component {component} is out of bounds for domain size {domainSize}.")

        # Sanity check: all points must be on the same side initially
        first_pos = self._constrained_points[0][self._component]
        for p in self._constrained_points:
            if (p[self._component] - boundaryPosition) * (first_pos - boundaryPosition) < -1e-15:
                raise ValueError("All constrained points must be on the same side of the boundary.")

        # Determine orientation: 1.0 if boundary is "above" particle, -1.0 if "below"
        self._orientation = 1.0 if boundaryPosition > first_pos else -1.0

        self._velocity = velocity
        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck

        if self._doProximityCheck:
            self._proximityDist = particle.getVolumeUndeformed() ** (1.0 / domainSize) * proximityFactor

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

    def updateConnectivity(self, model):
        """
        Updates the constraint connectivity.
        In explicit, this caches the global indices of the nodes for fast access.
        """
        # Update current coordinates of constrained points (particle may have moved)
        self._constrained_points = self._getConstraintLocations()

        # the nodes of the particle's kernel functions, and where they sit in the constraint's block
        localIndexOfNode = {}
        [self._localNodes] = locateTheNodesOfParticles([self._particle.kernelFunctions], localIndexOfNode)
        nodes = list(localIndexOfNode)
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        # within reach if one of the constrained points is close enough to the boundary
        self.withinReach = not self._doProximityCheck or any(
            abs(p[self._component] - self._boundaryPosition) <= self._proximityDist for p in self._constrained_points
        )

        if not self.active:
            self.reactionForce = 0.0

        return hasChanged

    def _springs(self, timeStep: TimeStep) -> SpringsOnPointsOfParticle:
        """The penalty springs of the boundary at the particle's current position: one of the penalty stiffness at every
        constrained point that penetrates the boundary, along the component the boundary is normal to, pushing the point
        back, :math:`f = -k\\, g` with the penetration :math:`g > 0`. The force (:meth:`computeKernelsExplicit`) and the
        bound of the stiffness (:meth:`computeBoundOfAbsoluteStiffnessTimes`) go through these springs.

        Parameters
        ----------
        timeStep
            The current time step, for the position of the moving boundary.

        Returns
        -------
        SpringsOnPointsOfParticle
            The springs; none if no point penetrates.
        """

        currentWallPos = self._boundaryPosition + self._velocity * timeStep.stepProgress

        # g > 0 is a penetration: the point is beyond the boundary on the side the orientation points to (1.0: the
        # boundary above the particle, -1.0: below it).
        points = np.array(self._constrained_points).reshape((len(self._constrained_points), -1))
        penetrations = self._orientation * (points[:, self._component] - currentWallPos)
        penetrating = penetrations > 0
        # along the orientation, with a negative force: the springs push the points back
        directions = np.zeros((penetrating.sum(), self._fieldSize))
        directions[:, self._component] = self._orientation
        return SpringsOnPointsOfParticle(
            # the interpolation vector with which the particle moves the point: the force is its transpose
            np.array(
                [self._particle.getAttachedPointInterpolationVector(p).flatten() for p in points[penetrating]]
            ).reshape((len(directions), len(self._localNodes))),
            directions,
            -self._penaltyParameter * penetrations[penetrating],
            np.full(len(directions), self._penaltyParameter),
        )

    def computeKernelsExplicit(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, timeStep: TimeStep):
        """Add the residual contribution of the penalty force: the negative of the force with which the boundary
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
            The current time step.
        """

        springs = self._springs(timeStep)
        addForcesOnPointsOfParticle(
            P.reshape(-1, self._fieldSize),
            self._localNodes,
            springs.interpolations,
            springs.forces[:, None] * springs.directions,
        )
        self.reactionForce = -springs.forces.sum()

    def computeBoundOfAbsoluteStiffnessTimes(
        self, U_np: np.ndarray, dU: np.ndarray, weights: np.ndarray, bound: np.ndarray, timeStep: TimeStep
    ):
        """Add the row sums of the absolute value of the stiffness of the penalty springs (:meth:`_springs`), weighted
        by column (see
        :func:`~edelweissmeshfree.constraints.base.pointsofparticles.addBoundOfAbsoluteStiffnessOfPointsOfParticle`).

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
        """

        addBoundOfAbsoluteStiffnessOfPointsOfParticle(bound, weights, self._localNodes, self._springs(timeStep))


def ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
    baseName: str,
    boundaryPosition: float,
    component: int,
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    field: str,
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
):
    """
    Factory function to create ParticleExplicitPenaltyContactCartesianBoundaryConstraint.
    """
    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraint = ParticlePenaltyCartesianBoundaryConstraintExplicit(
                    name,
                    p,
                    component,
                    boundaryPosition,
                    model,
                    location="face",
                    faceIDs=faceID,
                    penaltyParameter=penaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
                constraints[name] = constraint
        return constraints

    elif isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraint = ParticlePenaltyCartesianBoundaryConstraintExplicit(
                name,
                p,
                component,
                boundaryPosition,
                model,
                location,
                faceIDs,
                vertexIDs,
                penaltyParameter=penaltyParameter,
                doProximityCheck=doProximityCheck,
                proximityFactor=proximityFactor,
            )
            constraints[name] = constraint
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")
