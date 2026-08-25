from collections.abc import Iterable

import numpy as np
from edelweissfe.elements.contactsurfaceelement import facetNormalAndMeasure
from edelweissfe.utils.facetcontactgeometry import line2ClosestPoint, tria3ClosestPoint
from scipy.spatial import cKDTree

from edelweissmeshfree.models.mpmmodel import MPMModel


class ContactFacetSurface:
    """A faceted master contact surface, shared by every particle contact constraint acting on it.

    The surface is a compound of flat contact facet elements -- Tria3 in 3D, Line2 in 2D, as
    produced by :func:`edelweissfe.generators.surfaceelementgenerator` from the boundary of a
    solid-element body. It owns the geometry that is *common* to all contacting particles: the
    facets' current coordinates, outward normals, and a k-d tree over their centroids serving as
    the broad phase of the contact search.

    Sharing matters for performance. A particle contact constraint is instanced per particle, so
    an impact case has as many constraints as there are boundary particles. Rebuilding the search
    tree inside each of them would make the per-increment cost of the contact search quadratic in
    the problem size; here it is rebuilt once per increment, and every constraint queries it.

    Facet coordinates are read from the ``displacement`` node field, so the surface follows a
    deforming master body automatically. Whether the contacting particles also push *back* on the
    facet nodes is the constraint's business, not the surface's -- see
    :class:`~edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit.ParticlePenaltyFacetContactExplicit`.

    Parameters
    ----------
    name
        The name of the surface.
    facetElements
        The contact facet elements forming the surface. All of them must have the same number of
        nodes, matching the domain size (3 nodes in 3D, 2 nodes in 2D).
    model
        The MPM model instance.
    flipNormals
        Reverse the facets' outward normals. The sense of "outward" follows the facets' node
        ordering, which the surface element generator fixes to point away from the source body;
        set this if a hand-built facet set is oriented the other way. Getting it wrong does not
        fail loudly -- it silently inverts which side counts as penetration.
    """

    def __init__(self, name: str, facetElements: Iterable, model: MPMModel, flipNormals: bool = False):
        self._name = name
        self._facetElements = list(facetElements)
        self._domainSize = model.domainSize
        self._flipNormals = flipNormals

        if not self._facetElements:
            raise ValueError(f"ContactFacetSurface '{name}': no facet elements were given.")

        nodeCounts = {len(facet.nodes) for facet in self._facetElements}
        if len(nodeCounts) != 1:
            raise ValueError(
                f"ContactFacetSurface '{name}': all facets must have the same number of nodes, "
                f"but found {sorted(nodeCounts)}."
            )

        nNodesPerFacet = nodeCounts.pop()
        expectedNodesPerFacet = 3 if self._domainSize == 3 else 2
        if nNodesPerFacet != expectedNodesPerFacet:
            raise ValueError(
                f"ContactFacetSurface '{name}': a {self._domainSize}D surface is made of "
                f"{expectedNodesPerFacet}-node facets, but the given facets have {nNodesPerFacet} nodes."
            )

        self._closestPointFunction = tria3ClosestPoint if self._domainSize == 3 else line2ClosestPoint

        self._nodesOfFacets = [facet.nodes for facet in self._facetElements]
        self._referenceCoordinates = np.array(
            [[node.coordinates for node in facet.nodes] for facet in self._facetElements]
        )

        nFacets = len(self._facetElements)
        self._currentCoordinates = self._referenceCoordinates.copy()
        self._normals = np.zeros((nFacets, self._domainSize))
        self._measures = np.zeros(nFacets)
        self._centroids = np.zeros((nFacets, self._domainSize))
        self._maxCircumradius = 0.0
        self._tree = None
        self._refreshedAtTime = None

    @property
    def name(self) -> str:
        """The name of the surface.

        Returns
        -------
        str
            The name.
        """
        return self._name

    @property
    def facetElements(self) -> list:
        """The facet elements forming the surface, in the order the facet indices refer to.

        Returns
        -------
        list
            The facet elements.
        """
        return self._facetElements

    @property
    def nFacets(self) -> int:
        """The number of facets forming the surface.

        Returns
        -------
        int
            The number of facets.
        """
        return len(self._facetElements)

    def nodesOfFacet(self, facetIndex: int) -> list:
        """The nodes of one facet, in its fixed local order.

        Parameters
        ----------
        facetIndex
            The index of the facet.

        Returns
        -------
        list
            The facet's nodes.
        """
        return self._nodesOfFacets[facetIndex]

    def currentCoordinatesOfFacet(self, facetIndex: int) -> np.ndarray:
        """The current coordinates of one facet's nodes, as of the last refresh.

        Parameters
        ----------
        facetIndex
            The index of the facet.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape ``(nNodesPerFacet, domainSize)``.
        """
        return self._currentCoordinates[facetIndex]

    def normalOfFacet(self, facetIndex: int) -> np.ndarray:
        """The outward unit normal of one facet, as of the last refresh.

        Parameters
        ----------
        facetIndex
            The index of the facet.

        Returns
        -------
        numpy.ndarray
            The outward unit normal.
        """
        return self._normals[facetIndex]

    def measureOfFacet(self, facetIndex: int) -> float:
        """The measure (area in 3D, length in 2D) of one facet, as of the last refresh.

        Parameters
        ----------
        facetIndex
            The index of the facet.

        Returns
        -------
        float
            The measure.
        """
        return float(self._measures[facetIndex])

    def refresh(self, model: MPMModel) -> None:
        """Recompute the facets' current coordinates, normals, measures and centroids, and rebuild
        the broad-phase search tree.

        Parameters
        ----------
        model
            The MPM model instance, whose ``displacement`` node field the facet coordinates are
            read from.
        """
        self._currentCoordinates = self._computeCurrentCoordinates(model)

        for i, coordinates in enumerate(self._currentCoordinates):
            normal, measure = facetNormalAndMeasure(coordinates)
            self._normals[i] = -normal if self._flipNormals else normal
            self._measures[i] = measure

        self._centroids = self._currentCoordinates.mean(axis=1)

        # The broad phase prunes on centroid distance, so a facet's own extent has to be added to
        # the query radius: a large facet whose centroid is far away can still have its surface
        # within reach of the query point. Using the largest circumradius over all facets keeps
        # the padding a single scalar, at the price of being conservative for the small facets.
        circumradii = np.linalg.norm(self._currentCoordinates - self._centroids[:, np.newaxis, :], axis=2).max(axis=1)
        self._maxCircumradius = float(circumradii.max())

        self._tree = cKDTree(self._centroids)
        self._refreshedAtTime = model.time

    def refreshIfStale(self, model: MPMModel) -> bool:
        """Refresh the surface unless it has already been refreshed at the model's current time.

        This is what makes the surface shareable: every constraint calls this at the start of its
        own connectivity update, and only the first call of an increment does the work.

        The model time is the staleness key, which assumes the master geometry does not change
        within one increment -- true for the explicit solvers, which move the configuration once
        per increment. Call :meth:`invalidate` to force the next call to refresh.

        Parameters
        ----------
        model
            The MPM model instance.

        Returns
        -------
        bool
            True if the surface was refreshed by this call, False if a cached refresh was reused.
        """
        if self._refreshedAtTime is not None and self._refreshedAtTime == model.time:
            return False

        self.refresh(model)
        return True

    def invalidate(self) -> None:
        """Drop the staleness key, so the next :meth:`refreshIfStale` refreshes unconditionally."""
        self._refreshedAtTime = None

    def _computeCurrentCoordinates(self, model: MPMModel) -> np.ndarray:
        """The facets' node coordinates in the current configuration, i.e. displaced by the
        ``displacement`` node field where that field covers them.

        Parameters
        ----------
        model
            The MPM model instance.

        Returns
        -------
        numpy.ndarray
            The coordinates, of shape ``(nFacets, nNodesPerFacet, domainSize)``.
        """
        displacementField = model.nodeFields.get("displacement")

        if displacementField is None or "U" not in displacementField:
            return self._referenceCoordinates.copy()

        indicesOfNodes = displacementField._indicesOfNodesInArray
        U = displacementField["U"]

        currentCoordinates = self._referenceCoordinates.copy()
        for i, nodes in enumerate(self._nodesOfFacets):
            for j, node in enumerate(nodes):
                index = indicesOfNodes.get(node)
                if index is not None:
                    currentCoordinates[i, j] += U[index]

        return currentCoordinates

    def queryClosestFacet(self, coordinates: np.ndarray, searchDistance: float) -> tuple:
        """Find the facet closest to a point: a k-d tree broad phase on the facet centroids,
        followed by an exact clamped closest-point projection onto the surviving candidates.

        The projection is clamped to each facet's closed domain (interior, edges, vertices), so
        there is no dead zone at facet seams and the returned weights are non-negative and sum to
        one -- which is what lets the contact force be distributed over the facet's nodes without
        any of them being pulled the wrong way.

        Parameters
        ----------
        coordinates
            The query point.
        searchDistance
            The maximum distance from the point to the surface for a facet to be returned. A point
            that has travelled further than this into the master body within a single increment is
            not detected at all (tunnelling), so this wants to be generous relative to the
            per-increment motion.

        Returns
        -------
        tuple
            The tuple containing:
                - The index of the closest facet, or None if none is within ``searchDistance``.
                - The projection weights on that facet's nodes, or None.
                - The facet's outward unit normal, or None.
        """
        candidates = self._tree.query_ball_point(coordinates, searchDistance + self._maxCircumradius)

        closestFacetIndex = None
        closestWeights = None
        closestDistance = np.inf

        for facetIndex in candidates:
            weights, distance = self._closestPointFunction(coordinates, *self._currentCoordinates[facetIndex])
            if distance < closestDistance:
                closestFacetIndex, closestWeights, closestDistance = facetIndex, weights, distance

        if closestFacetIndex is None or closestDistance > searchDistance:
            return None, None, None

        return closestFacetIndex, closestWeights, self._normals[closestFacetIndex]
