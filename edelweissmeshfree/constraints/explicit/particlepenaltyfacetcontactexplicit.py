from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.constraints.contactfacetsurface import ContactFacetSurface
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


def resolveParticleConstraintLocations(
    particle: BaseParticle,
    location: str,
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
) -> list[np.ndarray]:
    """The coordinates of the points of a particle at which a contact constraint is evaluated.

    ``ParticlePenaltyContactImplicitSurfaceConstraintExplicit`` carries an equivalent block inline;
    it can be migrated onto this function when it is next touched.

    Parameters
    ----------
    particle
        The particle to evaluate.
    location
        Where on the particle the constraint acts: "center", "face", or "vertex".
    faceIDs
        The face ids to use if location is "face".
    vertexIDs
        The vertex ids to use if location is "vertex".

    Returns
    -------
    list[numpy.ndarray]
        The coordinates of the constrained points.
    """
    if location == "center":
        return [particle.getCenterCoordinates()]

    if location == "face":
        if faceIDs is None:
            raise ValueError("faceIDs must be specified when location is 'face'.")
        ids = [faceIDs] if isinstance(faceIDs, int) else faceIDs
        return [particle.getFaceCoordinates(faceID) for faceID in ids]

    if location == "vertex":
        if vertexIDs is None:
            raise ValueError("vertexIDs must be specified when location is 'vertex'.")
        ids = [vertexIDs] if isinstance(vertexIDs, int) else vertexIDs
        vertices = particle.getVertexCoordinates()
        return [vertices[vertexID] for vertexID in ids]

    raise ValueError(f"Unknown constraint location '{location}'; expected 'center', 'face' or 'vertex'.")


class ParticlePenaltyFacetContactExplicit(MPMConstraintBase):
    """Penalty contact between a particle and a faceted master surface, for EXPLICIT simulations.

    This is the faceted counterpart of
    :class:`~edelweissmeshfree.constraints.explicit.particlepenaltyrigidbodycontactexplicit.ParticlePenaltyContactImplicitSurfaceConstraintExplicit`:
    where that one needs the master body to be expressible as a closed-form signed distance
    function, this one takes the master surface as a mesh of flat facets, so the impact geometry
    can be an arbitrary meshed shape -- an anvil, a die, a projectile nose -- imported or generated
    like any other body. ``penaltyParameter`` has the same meaning in both, so one is a drop-in
    replacement for the other.

    Each constrained point of the particle is projected onto the closest facet, giving a signed
    gap along that facet's outward normal,

    .. math:: g = \\bar{n} \\cdot \\left( x_s - \\sum_i w_i \\, x_i \\right),

    negative in penetration, and a repulsive force :math:`-k \\, g \\, \\bar{n}` that is scattered
    onto the particle's kernel nodes through the particle's interpolation vector. The projection
    (facet, weights, normal) is frozen for the increment when the connectivity is updated, which is
    where the search cost is paid; the force evaluation is then pure arithmetic.

    With ``twoWayCoupling`` off, the master side is treated as prescribed: its geometry is read
    every increment, so the contact follows a master body that moves or deforms, but no reaction is
    applied back onto the facet nodes. That is what a rigid anvil or a driven punch wants, and it
    keeps the constraint's dof footprint down to the particle's own kernel nodes.

    With ``twoWayCoupling`` on, the reaction :math:`+f_n w_i \\bar{n}` is applied to the facet's
    nodes as well. Since the projection weights are non-negative and sum to one, the pair is an
    exact action-reaction pair and linear momentum is conserved. This requires the master elements
    to carry stiffness and inertia in the explicit solver, which
    :class:`~edelweissmeshfree.solvers.explicitmultiphysicssolver.ExplicitMultiphysicsSolver`
    assembles; pointing it at facets whose nodes belong to no element would push massless nodes.

    Parameters
    ----------
    name
        The name of the constraint.
    particle
        The particle to which the constraint is applied.
    surface
        The faceted master surface. Share one instance across all particles contacting the same
        surface -- it holds the search tree.
    model
        The MPM model instance.
    location
        Where to apply the constraint on the particle: "center", "face", or "vertex".
    faceIDs
        The face ids to apply the constraint at if location is "face".
    vertexIDs
        The vertex ids to apply the constraint at if location is "vertex".
    penaltyParameter
        The penalty stiffness, as a force per unit penetration of one constrained point. Higher
        values reduce penetration, but lower the stable time step of the explicit integration.
    searchDistance
        The distance from the surface within which a facet is assigned to a constrained point. If
        None, it is taken as the particle's size times ``proximityFactor``. A point that penetrates
        deeper than this within one increment tunnels through undetected.
    proximityFactor
        Multiplier on the particle size for the default ``searchDistance``.
    twoWayCoupling
        Apply the reaction to the master facet's nodes as well, so a deformable master body is
        pushed back by the contact. Leave off for a rigid or kinematically driven master.
    """

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        surface: ContactFacetSurface,
        model: MPMModel,
        location: str = "center",
        faceIDs: list[int] | int = None,
        vertexIDs: list[int] | int = None,
        penaltyParameter: float = 1e5,
        searchDistance: float = None,
        proximityFactor: float = 2.0,
        twoWayCoupling: bool = False,
    ):
        self._name = name
        self._field = "displacement"
        self._domainSize = model.domainSize
        self._fieldSize = getFieldSize(self._field, self._domainSize)

        self._particle = particle
        self._surface = surface
        self._penaltyParameter = penaltyParameter
        self._twoWayCoupling = twoWayCoupling

        self._location = location
        self._faceIDs = faceIDs
        self._vertexIDs = vertexIDs

        particleSize = particle.getVolumeUndeformed() ** (1.0 / self._domainSize)
        self._searchDistance = searchDistance if searchDistance is not None else particleSize * proximityFactor

        self._constrainedPoints = self._getConstraintLocations()
        self._assignedFacets = [(None, None, None)] * len(self._constrainedPoints)
        self._gaps = np.zeros(len(self._constrainedPoints))

        self._particleNodes = []
        self._nodes = []
        self._facetNodeOffsets = [None] * len(self._constrainedPoints)

        self.reactionForce = 0.0
        self.isActive = False

    def _getConstraintLocations(self) -> list[np.ndarray]:
        """The current coordinates of this constraint's constrained points on the particle.

        Returns
        -------
        list[numpy.ndarray]
            The coordinates of the constrained points.
        """
        return resolveParticleConstraintLocations(self._particle, self._location, self._faceIDs, self._vertexIDs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return [[self._field]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list:
        return []

    @property
    def active(self) -> bool:
        return self.isActive

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def getGaps(self) -> np.ndarray:
        """The signed gaps of the constrained points as of the last force evaluation, negative in
        penetration. Points with no facet assigned report a zero gap.

        Returns
        -------
        numpy.ndarray
            The gaps, ordered like the constrained points.
        """
        return self._gaps

    def updateConnectivity(self, model: MPMModel) -> bool:
        """Refresh the master surface, re-project every constrained point onto it, and refresh the
        grid nodes the particle currently reaches.

        Parameters
        ----------
        model
            The MPM model instance.

        Returns
        -------
        bool
            True if the contribution to the global system changed, False otherwise.
        """
        self._surface.refreshIfStale(model)

        self._constrainedPoints = self._getConstraintLocations()

        wasActive = self.isActive
        self.isActive = False

        self._assignedFacets = []
        for coordinates in self._constrainedPoints:
            assignment = self._surface.queryClosestFacet(coordinates, self._searchDistance)
            self._assignedFacets.append(assignment)
            if assignment[0] is not None:
                self.isActive = True

        hasChanged = self.isActive != wasActive

        # Refresh the grid nodes currently influenced by the particle. While the constraint is
        # inactive it claims no dofs at all, so an inactive constraint does not hold the dof
        # manager hostage to a particle whose support keeps changing.
        self._particleNodes = [kf.node for kf in self._particle.kernelFunctions] if self.isActive else []

        newNodes = list(self._particleNodes)
        self._facetNodeOffsets = [None] * len(self._constrainedPoints)

        if self._twoWayCoupling and self.isActive:
            # Each master node appears once, whatever number of constrained points project onto it.
            # A repeated node would be silently dropped rather than accumulated: the solver scatters
            # this block with ``P[c] += Pc``, and NumPy fancy indexing does not add duplicates up.
            localIndexOfNode = {}
            for i, (facetIndex, _, _) in enumerate(self._assignedFacets):
                if facetIndex is None:
                    continue
                offsets = []
                for node in self._surface.nodesOfFacet(facetIndex):
                    localIndex = localIndexOfNode.get(node)
                    if localIndex is None:
                        localIndex = len(newNodes)
                        localIndexOfNode[node] = localIndex
                        newNodes.append(node)
                    offsets.append(localIndex * self._fieldSize)
                self._facetNodeOffsets[i] = offsets

        if newNodes != self._nodes:
            hasChanged = True
        self._nodes = newNodes

        if not self.isActive:
            self.reactionForce = 0.0
            self._gaps = np.zeros(len(self._constrainedPoints))

        return hasChanged

    def applyConstraint(self, PExt: np.ndarray, timeStep: TimeStep):
        """Add the penalty contact forces to this constraint's block of the external force vector.

        Parameters
        ----------
        PExt
            This constraint's block of the global external force vector, of size :attr:`nDof`.
        timeStep
            The current time step. Unused; the penalty force depends on the configuration only.
        """
        if not self.isActive:
            return

        self._gaps = np.zeros(len(self._constrainedPoints))
        totalReaction = 0.0

        for i, (coordinates, (facetIndex, weights, normal)) in enumerate(
            zip(self._constrainedPoints, self._assignedFacets)
        ):
            if facetIndex is None:
                continue

            closestPoint = weights @ self._surface.currentCoordinatesOfFacet(facetIndex)
            gap = float(normal.dot(coordinates - closestPoint))
            self._gaps[i] = gap

            if gap >= 0.0:
                continue

            # Repulsive: gap is negative in penetration, so this points along the outward normal.
            forceVector = -self._penaltyParameter * gap * normal

            N = self._particle.getInterpolationVector(coordinates).flatten()
            self._scatterOntoParticle(PExt, N, forceVector)

            if self._facetNodeOffsets[i] is not None:
                self._scatterOntoMaster(PExt, self._facetNodeOffsets[i], weights, forceVector)

            totalReaction += -self._penaltyParameter * gap

        self.reactionForce = totalReaction

    def _scatterOntoParticle(self, PExt: np.ndarray, N: np.ndarray, forceVector: np.ndarray):
        """Distribute a force acting at a point of the particle onto the particle's kernel nodes.

        Writes into the leading, particle-owned block of ``PExt``, so that appending the master
        side's nodes to :attr:`nodes` later does not disturb this scatter.

        Parameters
        ----------
        PExt
            This constraint's block of the global external force vector.
        N
            The particle's interpolation vector at the point the force acts at.
        forceVector
            The force acting at that point.
        """
        particleBlock = PExt[: len(self._particleNodes) * self._fieldSize]

        for d in range(self._domainSize):
            particleBlock[d :: self._fieldSize] += N * forceVector[d]

    def _scatterOntoMaster(self, PExt: np.ndarray, offsets: list[int], weights: np.ndarray, forceVector: np.ndarray):
        """Distribute the reaction to the contact force over the master facet's nodes.

        The weights are the projection's and sum to one, so this takes exactly as much as
        :meth:`_scatterOntoParticle` gave, and linear momentum is conserved. Written one node at a
        time with plain slices rather than in one fancy-indexed shot, so that two constrained points
        sharing a master node accumulate instead of overwriting.

        Parameters
        ----------
        PExt
            This constraint's block of the global external force vector.
        offsets
            The start of each facet node's entries within ``PExt``, in facet node order.
        weights
            The projection weights on the facet's nodes.
        forceVector
            The force acting on the particle, whose negative is shared out here.
        """
        for offset, weight in zip(offsets, weights):
            PExt[offset : offset + self._domainSize] -= weight * forceVector


def ParticlePenaltyFacetContactExplicitFactory(
    baseName: str,
    surface: ContactFacetSurface,
    particleCollection: Iterable[BaseParticle],
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    searchDistance: float = None,
    proximityFactor: float = 2.0,
    twoWayCoupling: bool = False,
):
    """Create one :class:`ParticlePenaltyFacetContactExplicit` per particle of a collection, all
    sharing the given master surface.

    Sharing the surface is the point of the factory: the search tree is built once per increment
    for the whole collection rather than once per particle.

    Parameters
    ----------
    baseName
        Base name for the constraints. A unique suffix is appended per particle.
    surface
        The faceted master surface all the constraints contact.
    particleCollection
        The particles to create constraints for.
    model
        The MPM model instance.
    location
        Where to apply the constraints on the particles: "center", "face", or "vertex".
    faceIDs
        The face ids to apply the constraints at if location is "face".
    vertexIDs
        The vertex ids to apply the constraints at if location is "vertex".
    penaltyParameter
        The penalty stiffness, as a force per unit penetration of one constrained point.
    searchDistance
        The search distance; if None, derived per particle from its size and ``proximityFactor``.
    proximityFactor
        Multiplier on the particle size for the default ``searchDistance``.
    twoWayCoupling
        Apply the reaction to the master facet's nodes as well.

    Returns
    -------
    dict
        The constraints, keyed by name.
    """
    if not isinstance(particleCollection, Iterable):
        raise TypeError("particleCollection must be an iterable of particles.")

    constraints = dict()

    for i, particle in enumerate(particleCollection):
        name = f"{baseName}_{i}"
        constraints[name] = ParticlePenaltyFacetContactExplicit(
            name,
            particle,
            surface,
            model,
            location,
            faceIDs,
            vertexIDs,
            penaltyParameter,
            searchDistance,
            proximityFactor,
            twoWayCoupling,
        )

    return constraints
