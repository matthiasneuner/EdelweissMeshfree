"""Unit tests for the faceted master contact surface shared by the particle contact constraints.

These cover the parts that fail silently rather than loudly: the sense of the facet normals (and
with it which side counts as penetration), the clamped projection off the edge of the surface, the
padding of the broad-phase query radius by the facets' own extent, and the refresh caching that
makes the surface shareable across many constraints.
"""

import numpy as np
import pytest
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.elements.contactsurfaceelement import (
    Line2ContactFacet,
    Tria3ContactFacet,
)
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.fieldvariable import FieldVariable

from edelweissmeshfree.constraints.contactfacetsurface import ContactFacetSurface
from edelweissmeshfree.models.mpmmodel import MPMModel


def _makeNode(label, coordinates):
    node = Node(label, np.array(coordinates, dtype=float))
    node.fields["displacement"] = FieldVariable(node, "displacement")
    return node


def _makeFacet(facetClass, elNumber, nodes, initialize=True):
    facet = facetClass(facetClass.__name__, elNumber)
    facet.setNodes(nodes)
    if initialize:
        # ContactFacetSurface itself needs nothing but the nodes; a real model initializes its
        # facets, so the fixtures do too, but the guard tests below build facets a real model
        # could never initialize.
        facet.initializeElement()
    return facet


@pytest.fixture
def unitSquareSurface():
    """Two Tria3 facets triangulating the unit square in the z=0 plane, outward normal +z."""

    model = MPMModel(3)

    nodes = {
        1: _makeNode(1, [0.0, 0.0, 0.0]),
        2: _makeNode(2, [1.0, 0.0, 0.0]),
        3: _makeNode(3, [1.0, 1.0, 0.0]),
        4: _makeNode(4, [0.0, 1.0, 0.0]),
    }
    model.nodes = nodes

    facets = [
        _makeFacet(Tria3ContactFacet, 1, [nodes[1], nodes[2], nodes[4]]),
        _makeFacet(Tria3ContactFacet, 2, [nodes[2], nodes[3], nodes[4]]),
    ]

    surface = ContactFacetSurface("theSurface", facets, model)
    surface.refresh(model)

    return surface, model, nodes


def _signedGap(surface, coordinates, searchDistance):
    """The signed gap the constraint would compute, or None if no facet is within reach."""

    facetIndex, weights, normal = surface.queryClosestFacet(np.array(coordinates, dtype=float), searchDistance)
    if facetIndex is None:
        return None
    closestPoint = weights @ surface.currentCoordinatesOfFacet(facetIndex)
    return float(normal.dot(np.array(coordinates, dtype=float) - closestPoint))


class TestGeometry:
    def test_normals_point_outward_and_measures_are_the_triangle_areas(self, unitSquareSurface):
        surface, _, _ = unitSquareSurface

        assert surface.nFacets == 2
        for facetIndex in range(surface.nFacets):
            np.testing.assert_allclose(surface.normalOfFacet(facetIndex), [0.0, 0.0, 1.0], atol=1e-14)
            assert surface.measureOfFacet(facetIndex) == pytest.approx(0.5)

    def test_flipped_normals_are_reversed(self, unitSquareSurface):
        _, model, nodes = unitSquareSurface

        facets = [
            _makeFacet(Tria3ContactFacet, 1, [nodes[1], nodes[2], nodes[4]]),
            _makeFacet(Tria3ContactFacet, 2, [nodes[2], nodes[3], nodes[4]]),
        ]
        flipped = ContactFacetSurface("flipped", facets, model, flipNormals=True)
        flipped.refresh(model)

        for facetIndex in range(flipped.nFacets):
            np.testing.assert_allclose(flipped.normalOfFacet(facetIndex), [0.0, 0.0, -1.0], atol=1e-14)

    def test_projection_weights_are_a_partition_of_unity(self, unitSquareSurface):
        surface, _, _ = unitSquareSurface

        _, weights, _ = surface.queryClosestFacet(np.array([0.2, 0.2, 0.1]), 0.5)

        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0.0).all()

    def test_gap_is_positive_outside_and_negative_in_penetration(self, unitSquareSurface):
        surface, _, _ = unitSquareSurface

        assert _signedGap(surface, [0.25, 0.25, 0.1], 0.5) == pytest.approx(0.1)
        assert _signedGap(surface, [0.25, 0.25, -0.05], 0.5) == pytest.approx(-0.05)

    def test_a_point_out_of_reach_is_not_assigned_a_facet(self, unitSquareSurface):
        surface, _, _ = unitSquareSurface

        assert surface.queryClosestFacet(np.array([0.25, 0.25, 5.0]), 0.5) == (None, None, None)

    def test_projection_past_the_edge_clamps_onto_the_boundary(self, unitSquareSurface):
        surface, _, _ = unitSquareSurface

        # 0.5 beyond the x=1 edge and 0.05 above the plane: the closest point of the closed
        # surface is on that edge, at a distance just over 0.5.
        beyondTheEdge = [1.5, 0.5, 0.05]

        assert surface.queryClosestFacet(np.array(beyondTheEdge), 0.5) == (None, None, None)

        facetIndex, weights, _ = surface.queryClosestFacet(np.array(beyondTheEdge), 1.0)
        assert facetIndex is not None
        assert (weights >= 0.0).all()
        # The gap is still measured along the facet normal, so it reports the height above the plane.
        assert _signedGap(surface, beyondTheEdge, 1.0) == pytest.approx(0.05)

    def test_a_facet_much_larger_than_the_search_distance_is_still_found(self):
        """The broad phase prunes on centroid distance, so it has to be padded by the facets' own
        extent -- without that, a point right above the corner of a large facet is missed."""

        model = MPMModel(3)
        nodes = [
            _makeNode(1, [0.0, 0.0, 0.0]),
            _makeNode(2, [100.0, 0.0, 0.0]),
            _makeNode(3, [0.0, 100.0, 0.0]),
        ]
        model.nodes = {n.label: n for n in nodes}

        surface = ContactFacetSurface("oneBigFacet", [_makeFacet(Tria3ContactFacet, 1, nodes)], model)
        surface.refresh(model)

        # Distance to the centroid is ~47, far outside a search distance of 0.1.
        facetIndex, _, _ = surface.queryClosestFacet(np.array([0.5, 0.5, 0.05]), 0.1)
        assert facetIndex == 0

    def test_line2_facets_in_2d(self):
        model = MPMModel(2)
        nodes = [_makeNode(1, [0.0, 0.0]), _makeNode(2, [1.0, 0.0]), _makeNode(3, [2.0, 0.0])]
        model.nodes = {n.label: n for n in nodes}

        facets = [
            _makeFacet(Line2ContactFacet, 1, [nodes[1], nodes[0]]),
            _makeFacet(Line2ContactFacet, 2, [nodes[2], nodes[1]]),
        ]
        surface = ContactFacetSurface("theLine", facets, model)
        surface.refresh(model)

        # Traversal from +x to -x puts the outward normal at +y.
        for facetIndex in range(surface.nFacets):
            np.testing.assert_allclose(surface.normalOfFacet(facetIndex), [0.0, 1.0], atol=1e-14)
            assert surface.measureOfFacet(facetIndex) == pytest.approx(1.0)

        assert _signedGap(surface, [0.5, 0.2], 0.5) == pytest.approx(0.2)
        assert _signedGap(surface, [1.5, -0.1], 0.5) == pytest.approx(-0.1)

    def test_an_empty_facet_set_is_rejected(self):
        with pytest.raises(ValueError, match="no facet elements"):
            ContactFacetSurface("empty", [], MPMModel(3))

    def test_facets_of_the_wrong_shape_for_the_domain_are_rejected(self):
        model = MPMModel(3)
        nodes = [_makeNode(i + 1, [float(i), 0.0, 0.0]) for i in range(3)]
        model.nodes = {n.label: n for n in nodes}

        line = _makeFacet(Line2ContactFacet, 1, nodes[:2], initialize=False)
        with pytest.raises(ValueError, match="3-node facets"):
            ContactFacetSurface("wrongShape", [line], model)

    def test_a_facet_set_of_mixed_shapes_is_rejected(self):
        model = MPMModel(3)
        nodes = [_makeNode(i + 1, [float(i), 0.0, 0.0]) for i in range(3)]
        model.nodes = {n.label: n for n in nodes}

        facets = [
            _makeFacet(Tria3ContactFacet, 1, nodes, initialize=False),
            _makeFacet(Line2ContactFacet, 2, nodes[:2], initialize=False),
        ]
        with pytest.raises(ValueError, match="same number of nodes"):
            ContactFacetSurface("mixed", facets, model)


class TestFollowsADeformingMaster:
    def test_facet_coordinates_follow_the_displacement_field(self, unitSquareSurface):
        surface, model, nodes = unitSquareSurface

        nodeSet = NodeSet("all", list(nodes.values()))
        displacementField = NodeField("displacement", getFieldSize("displacement", 3), nodeSet)
        U = displacementField.createFieldValueEntry("U")
        U[:, 2] = 0.25  # lift the whole surface
        model.nodeFields = {"displacement": displacementField}

        surface.refresh(model)

        for facetIndex in range(surface.nFacets):
            np.testing.assert_allclose(surface.currentCoordinatesOfFacet(facetIndex)[:, 2], 0.25, atol=1e-14)
            np.testing.assert_allclose(surface.normalOfFacet(facetIndex), [0.0, 0.0, 1.0], atol=1e-14)

        # A point that was 0.1 clear of the reference surface is now 0.15 inside the lifted one.
        assert _signedGap(surface, [0.25, 0.25, 0.1], 0.5) == pytest.approx(-0.15)

    def test_a_node_missing_from_the_displacement_field_keeps_its_reference_position(self, unitSquareSurface):
        surface, model, nodes = unitSquareSurface

        # Only two of the four nodes carry the field; the others must not be silently shifted.
        nodeSet = NodeSet("some", [nodes[1], nodes[2]])
        displacementField = NodeField("displacement", getFieldSize("displacement", 3), nodeSet)
        U = displacementField.createFieldValueEntry("U")
        U[:, 2] = 1.0
        model.nodeFields = {"displacement": displacementField}

        surface.refresh(model)

        coordinates = surface.currentCoordinatesOfFacet(0)
        np.testing.assert_allclose(coordinates[0], [0.0, 0.0, 1.0], atol=1e-14)  # node 1, displaced
        np.testing.assert_allclose(coordinates[1], [1.0, 0.0, 1.0], atol=1e-14)  # node 2, displaced
        np.testing.assert_allclose(coordinates[2], [0.0, 1.0, 0.0], atol=1e-14)  # node 4, untouched


class TestRefreshCaching:
    def test_refresh_happens_once_per_model_time(self, unitSquareSurface):
        surface, model, _ = unitSquareSurface

        model.time = 1.0
        assert surface.refreshIfStale(model) is True
        assert surface.refreshIfStale(model) is False

        model.time = 2.0
        assert surface.refreshIfStale(model) is True
        assert surface.refreshIfStale(model) is False

    def test_invalidate_forces_the_next_refresh(self, unitSquareSurface):
        surface, model, _ = unitSquareSurface

        model.time = 1.0
        surface.refreshIfStale(model)

        surface.invalidate()
        assert surface.refreshIfStale(model) is True
