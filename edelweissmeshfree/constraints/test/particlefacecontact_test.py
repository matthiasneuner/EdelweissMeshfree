# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""The contact of particle faces with a master surface of finite element faces: EdelweissFE's contact constraint
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`) with
:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`. The particles
are stand-ins with the interpolation of a real particle linearized about its center (affine in the point); the master
is a block of plane strain elements with its generated contact facets."""

import numpy as np
import pytest
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    ContactState,
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.generators.planerectquad import Generator as PlaneRectQuadGenerator
from edelweissfe.generators.planerectquad import PlaneRectQuadSchema
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.points.node import Node
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.residualtangentcheck import (
    finiteDifferenceTangent,
    residualContributionAndTangent,
)

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
from edelweissmeshfree.constraints.test._residualtangent import ParticleStandIn
from edelweissmeshfree.models.mpmmodel import MPMModel

PENALTY = 1e3
SEARCH = 0.2
SIZE = 0.25


class JournalStandIn:
    """Collects what the constraint reports."""

    def __init__(self):
        self.messages = []

    def message(self, text, senderIdentification, level=1):
        self.messages.append(text)


def _masterBlock(model: MPMModel, journal, height: float = 1.0, withBottom: bool = False) -> ElementSet:
    """A block of 4 x 1 plane strain elements below y = 0, of the given height, and the contact facets of its top (and
    bottom) surface: the master."""

    with model.topology.changes():
        PlaneRectQuadGenerator(
            "block",
            model,
            journal,
            configuration=PlaneRectQuadSchema(x0=0.0, y0=-height, length=1.0, h=height, nX=4, nY=1, elType="CPE4"),
        )
        facets = list(
            model.elementSets[buildContactFacets(model, "block_top", "top", "midside", "facetConsistent", journal)[0]]
        )
        if withBottom:
            facets += list(model.elementSets[buildContactFacets(model, "block_bottom", "bottom", "midside", "facetConsistent", journal)[0]])  # fmt: skip
    return ElementSet("master", facets)


def _particle(x0: float, y0: float, number: int = 1, angle: float = 0.0) -> ParticleStandIn:
    """A square particle stand-in of the side SIZE with its lower left vertex at (x0, y0), rotated by ``angle`` about
    that vertex, on four kernel nodes around it."""

    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    origin = np.array([x0, y0])
    vertices = origin + np.array([[0.0, 0.0], [SIZE, 0.0], [SIZE, SIZE], [0.0, SIZE]]) @ rotation.T
    center = vertices.mean(axis=0)
    offsets = np.array([[-0.7, -0.6], [0.8, -0.7], [0.7, 0.9], [-0.6, 0.6]]) * SIZE
    nodes = [Node(100 * number + i, center + offset) for i, offset in enumerate(offsets)]
    particle = ParticleStandIn(nodes, center, vertices)
    particle.number = number
    return particle


def _move(particle: ParticleStandIn, offset):
    particle.vertices = particle.vertices + np.asarray(offset, dtype=float)
    particle.center = particle.center + np.asarray(offset, dtype=float)


def _contact(
    model,
    particles,
    master,
    journal=None,
    nQuadraturePoints=1,
    contactType="linear",
    searchDistance=SEARCH,
    initialOverlap="pushApart",
):
    """The contact of the bottom faces (face 1) of the particles with the master, connected."""

    contact = SurfaceContact(
        "contact",
        model,
        ContactPointsOnParticleFaces([(particle, 1) for particle in particles], nQuadraturePoints),
        master,
        journal or JournalStandIn(),
        configuration=SurfaceToDeformableSurfacePenaltySchema(
            penalty=PENALTY, contactType=contactType, searchDistance=searchDistance, initialOverlap=initialOverlap
        ),
    )
    contact.updateConnectivity(model)
    return contact


def _residual(contact) -> np.ndarray:
    P = np.zeros(contact.nDof)
    contact.computeKernelsExplicit(np.zeros(contact.nDof), np.zeros(contact.nDof), P, None)
    return P


@pytest.mark.parametrize("nQuadraturePoints", [1, 2])
def test_a_face_pressed_into_the_master_transmits_the_penalty_times_its_area_times_the_overlap(nQuadraturePoints):
    """A face overlapping a flat master uniformly by d: the pressure is the penalty times d at every point, the force the
    pressure times the face's length, whatever the rule; it pushes the particle up and the master down, equal and
    opposite."""

    model = MPMModel(2)
    master = _masterBlock(model, JournalStandIn())
    particle = _particle(0.3, -0.02)
    contact = _contact(model, [particle], master, nQuadraturePoints=nQuadraturePoints)

    assert (contact.stateOfPoints == ContactState.IN_CONTACT).all()
    P = _residual(contact).reshape(-1, 2)

    np.testing.assert_allclose(contact.getNormalPressures(), PENALTY * 0.02, rtol=1e-12)
    np.testing.assert_allclose(contact.contactForceOnSlave, [0.0, PENALTY * 0.02 * SIZE], atol=1e-12)
    np.testing.assert_allclose(contact.contactForceOnSlave + contact.contactForceOnMaster, 0.0, atol=1e-12)
    # the residual contribution is the negative of the force: on the four kernel nodes of the particle first
    np.testing.assert_allclose(P[:4].sum(axis=0), -contact.contactForceOnSlave, atol=1e-12)
    np.testing.assert_allclose(P[4:].sum(axis=0), -contact.contactForceOnMaster, atol=1e-12)
    assert contact.penaltyEnergy == pytest.approx(0.5 * PENALTY * SIZE * 0.02**2)
    assert contact.deepestPenetration == pytest.approx(0.02)


@pytest.mark.parametrize("contactType", ["linear", "quadratic"])
def test_the_tangent_is_the_derivative_of_the_residual_contribution(contactType):
    """``K == dP/dU`` by central differences, for a tilted face of two particles astride two elements, whose points
    stay in contact; the particle points move with the increment, the master with the total displacement."""

    model = MPMModel(2)
    master = _masterBlock(model, JournalStandIn())
    particles = [_particle(0.12, -0.03, 1, angle=0.05), _particle(0.40, -0.02, 2, angle=-0.03)]
    contact = _contact(model, particles, master, nQuadraturePoints=2, contactType=contactType)
    assert (contact.stateOfPoints == ContactState.IN_CONTACT).all()

    rng = np.random.default_rng(1)
    dU = 1e-3 * rng.standard_normal(contact.nDof)
    U = dU.copy()

    P, K = residualContributionAndTangent(contact, U, dU)
    dP_dU = finiteDifferenceTangent(contact, U, dU, h=1e-8)
    assert np.abs(P).max() > 0.0
    np.testing.assert_allclose(K, dP_dU, atol=1e-6 * np.abs(K).max())
    np.testing.assert_allclose(K, K.T, atol=1e-12 * np.abs(K).max())


def test_an_overlap_at_the_first_update_is_contact():
    """The bodies as the model gives them: a face that overlaps the master at the first update (e.g. by the faceting of
    a curved surface) is in contact and is pushed out."""

    model = MPMModel(2)
    contact = _contact(model, [_particle(0.3, -1e-4)], _masterBlock(model, JournalStandIn()))
    assert contact.stateOfPoints[0] == ContactState.IN_CONTACT
    _residual(contact)
    assert contact.numberOfPointsInContact == 1


def test_an_overlap_at_the_first_update_can_be_ramped_out():
    """With ``initialOverlapRampTime``, a tilted face that overlaps the master at the first update starts without a
    force -- the overlap is measured for each point as the evaluation measures it, from the particle -- and carries the
    full force once the ramp time has passed."""

    def at(time: float) -> TimeStep:
        return TimeStep(1, 0.0, 0.0, 0.0, time, time)

    forces = {}
    for rampTime in (0.0, 1e-3):
        model = MPMModel(2)
        particles = [_particle(0.12, -0.03, 1, angle=0.05), _particle(0.40, -0.02, 2, angle=-0.03)]
        contact = SurfaceContact(
            "contact",
            model,
            ContactPointsOnParticleFaces([(particle, 1) for particle in particles], 2),
            _masterBlock(model, JournalStandIn()),
            JournalStandIn(),
            configuration=SurfaceToDeformableSurfacePenaltySchema(
                penalty=PENALTY, searchDistance=SEARCH, initialOverlapRampTime=rampTime
            ),
        )
        contact.updateConnectivity(model)
        assert (contact.stateOfPoints == ContactState.IN_CONTACT).all()
        for time in (0.0, 2e-3):
            contact.computeKernelsExplicit(
                np.zeros(contact.nDof), np.zeros(contact.nDof), np.zeros(contact.nDof), at(time)
            )
            forces[rampTime, time] = contact.contactForceOnSlave

    np.testing.assert_allclose(forces[1e-3, 0.0], 0.0, atol=1e-12 * np.abs(forces[0.0, 0.0]).max())
    assert np.abs(forces[0.0, 0.0]).max() > 0.0
    np.testing.assert_array_equal(forces[1e-3, 2e-3], forces[0.0, 2e-3])


def test_an_adjusted_overlap_at_the_first_update_is_the_state_at_rest():
    """With ``initialOverlap=adjust``, tilted faces of particles that overlap the master at the first update (flat faces
    standing for a curved surface) rest there without a force; pressed further in, they carry the force of the further
    approach alone -- what the same faces pushed apart carry more than at the start, the law being linear -- with the
    tangent the derivative of the residual contribution. A checkpoint carries the offsets."""

    def residualAfter(contact, approach: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dU = np.zeros(contact.nDof)
        dU[: contact._nSlaveDof] = np.tile(approach, contact._nSlaveDof // 2)  # the particles' kernel nodes
        return residualContributionAndTangent(contact, dU, dU)

    residuals, contacts = {}, {}
    for initialOverlap in ("pushApart", "adjust"):
        model = MPMModel(2)
        particles = [_particle(0.12, -0.03, 1, angle=0.05), _particle(0.40, -0.02, 2, angle=-0.03)]
        contact = _contact(
            model, particles, _masterBlock(model, JournalStandIn()), nQuadraturePoints=2, initialOverlap=initialOverlap
        )
        assert (contact.stateOfPoints == ContactState.IN_CONTACT).all()
        for approach in ((0.0, 0.0), (0.0, -0.01)):
            residuals[initialOverlap, approach] = residualAfter(contact, np.array(approach))[0]
        contacts[initialOverlap] = contact

    adjusted = contacts["adjust"]
    assert adjusted._initialOverlap.min() < -0.02
    np.testing.assert_array_equal(residuals["adjust", (0.0, 0.0)], 0.0)
    pushedFurther = residuals["pushApart", (0.0, -0.01)] - residuals["pushApart", (0.0, 0.0)]
    assert np.abs(pushedFurther).max() > 0.0
    np.testing.assert_allclose(
        residuals["adjust", (0.0, -0.01)], pushedFurther, atol=1e-12 * np.abs(pushedFurther).max()
    )

    dU = 1e-3 * np.random.default_rng(5).standard_normal(adjusted.nDof)
    dU[: adjusted._nSlaveDof] += np.tile([0.0, -0.01], adjusted._nSlaveDof // 2)
    K = residualContributionAndTangent(adjusted, dU, dU)[1]
    np.testing.assert_allclose(K, finiteDifferenceTangent(adjusted, dU, dU, h=1e-8), atol=1e-6 * np.abs(K).max())

    model = MPMModel(2)
    particles = [_particle(0.12, -0.03, 1, angle=0.05), _particle(0.40, -0.02, 2, angle=-0.03)]
    master = _masterBlock(model, JournalStandIn())
    checkpointed = _contact(model, particles, master, nQuadraturePoints=2, initialOverlap="adjust")
    restored = _contact(model, particles, master, nQuadraturePoints=2, initialOverlap="adjust", searchDistance=1e-9)
    restored.setRestartData(checkpointed.getRestartData())
    restored.updateConnectivity(model)  # locates the restored points; not the first search: nothing recorded anew
    np.testing.assert_array_equal(restored._initialOverlap, checkpointed._initialOverlap)
    np.testing.assert_array_equal(residualAfter(restored, np.array([0.0, -0.01]))[0], residuals["adjust", (0.0, -0.01)])


def test_a_point_that_tunnels_through_the_search_distance_gets_no_force_and_is_reported():
    journal = JournalStandIn()
    model = MPMModel(2)
    particle = _particle(0.3, 0.5)
    contact = _contact(model, [particle], _masterBlock(model, journal), journal)
    assert contact.stateOfPoints[0] == ContactState.FREE and not contact.withinReach

    _move(particle, [0.0, -0.55])
    contact.updateConnectivity(model)
    assert contact.stateOfPoints[0] == ContactState.FOUND_INSIDE
    assert contact.numberOfPointsFoundInside == 1
    assert not contact.withinReach and contact.nodes == []
    assert any("found inside" in message for message in journal.messages)

    # seen outside again, it may come into contact as any other point
    _move(particle, [0.0, 0.1])
    contact.updateConnectivity(model)
    assert contact.stateOfPoints[0] == ContactState.APPROACHING


def test_a_point_deep_inside_a_thin_master_is_pushed_back_the_way_it_came():
    """The closest facet of a point that is deeper than half the thickness of a thin master is on its far side, whose
    normal points the wrong way: a point in contact is not paired with a facet that faces against its normal."""

    model = MPMModel(2)
    particle = _particle(0.3, 0.05)
    contact = _contact(model, [particle], _masterBlock(model, JournalStandIn(), height=0.2, withBottom=True))

    for y in np.linspace(-0.02, -0.15, 14):
        _move(particle, [0.0, y - particle.vertices[0, 1]])
        contact.updateConnectivity(model)
        assert contact.stateOfPoints[0] == ContactState.IN_CONTACT

    _residual(contact)
    assert contact.deepestPenetration == pytest.approx(0.15)
    np.testing.assert_allclose(contact.contactForceOnSlave, [0.0, PENALTY * 0.15 * SIZE], atol=1e-9)


def test_a_point_in_contact_slides_on_across_facets_and_elements():
    model = MPMModel(2)
    particle = _particle(0.02, -0.01)
    contact = _contact(model, [particle], _masterBlock(model, JournalStandIn()))
    firstFacet = contact._assignedFacetIdx[0]

    for x in np.linspace(0.05, 0.7, 14):
        _move(particle, [x - particle.vertices[0, 0], 0.0])
        contact.updateConnectivity(model)
        assert contact.stateOfPoints[0] == ContactState.IN_CONTACT

    assert contact.facetElements[contact._assignedFacetIdx[0]].parentFaceNodes != (
        contact.facetElements[firstFacet].parentFaceNodes
    )
    _residual(contact)
    assert contact.deepestPenetration == pytest.approx(0.01)


def test_a_point_that_leaves_is_released_and_out_of_reach_takes_the_constraint_out_of_reach():
    model = MPMModel(2)
    particle = _particle(0.3, -0.01)
    contact = _contact(model, [particle], _masterBlock(model, JournalStandIn()))
    _residual(contact)

    _move(particle, [0.0, 0.02])
    contact.updateConnectivity(model)
    _residual(contact)
    assert contact.stateOfPoints[0] == ContactState.APPROACHING
    np.testing.assert_array_equal(contact.contactForceOnSlave, 0.0)
    assert contact.penaltyEnergy == 0.0

    _move(particle, [0.0, 0.5])
    assert contact.updateConnectivity(model)
    assert contact.stateOfPoints[0] == ContactState.FREE
    assert not contact.withinReach and contact.nodes == []


def test_the_restart_continues_the_pairing():
    """A contact restored from a checkpoint continues the pairing: a point in contact stays in contact, also when the
    search distance would not find it anew. The checkpoint is read before the particles are connected to their kernel
    functions (a resumed run connects them at its first update of the connectivity), so the points are located then."""

    model = MPMModel(2)
    master = _masterBlock(model, JournalStandIn())
    particle = _particle(0.3, -0.01)
    contact = _contact(model, [particle], master)
    forces = _residual(contact)

    restored = _contact(model, [particle], master, searchDistance=1e-9)
    assert restored.stateOfPoints[0] == ContactState.FREE
    kernelFunctions, particle.kernelFunctions = particle.kernelFunctions, None
    restored.setRestartData(contact.getRestartData())
    assert restored.stateOfPoints[0] == ContactState.IN_CONTACT

    particle.kernelFunctions = kernelFunctions
    restored.updateConnectivity(model)
    assert restored.stateOfPoints[0] == ContactState.IN_CONTACT
    np.testing.assert_array_equal(_residual(restored), forces)


def test_faces_of_particles_other_than_quadrilaterals_and_hexahedra_are_refused():
    particle = ParticleStandIn([Node(1, np.zeros(2))], np.zeros(2), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="quadrilateral"):
        ContactPointsOnParticleFaces([(particle, 1)])


@pytest.mark.parametrize("faceID", [0, -1, 5])
def test_face_ids_outside_the_abaqus_numbering_are_refused(faceID):
    """Face IDs count from 1 (Abaqus): an ID 0, from a numbering that counts from 0, must not wrap around to the last
    face, and one beyond the last face must not reach an index error."""

    with pytest.raises(ValueError, match="1 to 4"):
        ContactPointsOnParticleFaces([(_particle(0.0, 0.0), 1), (_particle(0.0, 0.0, 2), faceID)])


def _tie(model, particles, master, nQuadraturePoints=2, searchDistance=SEARCH):
    """The tie (``type=tied``) of the bottom faces of the particles to the master, connected."""

    return _contact(
        model, particles, master, nQuadraturePoints=nQuadraturePoints, contactType="tied", searchDistance=searchDistance
    )


@pytest.mark.parametrize("y0", [0.02, 0.0, -0.02])
def test_a_tied_face_with_a_gap_or_an_overlap_is_at_rest_and_pulled_back_in_every_direction(y0):
    """The gap (or overlap) of a particle face at the pairing is the state at rest: no force. Moved by ``d`` with its
    kernel nodes, the face is pulled back with the traction ``-k d``, away from the master and along it alike."""

    model = MPMModel(2)
    master = _masterBlock(model, JournalStandIn())
    particle = _particle(0.3, y0)
    tie = _tie(model, [particle], master)
    assert (tie.stateOfPoints == ContactState.TIED).all()

    np.testing.assert_allclose(_residual(tie), 0.0, atol=1e-12)
    for d in ([0.0, 0.01], [0.0, -0.01], [0.01, 0.0], [-0.01, 0.005]):
        dU = np.zeros(tie.nDof)
        dU[: 2 * len(particle.kernelFunctions)] = np.tile(d, len(particle.kernelFunctions))  # the particle's nodes
        P = np.zeros(tie.nDof)
        tie.computeKernelsExplicit(dU, dU, P, None)
        np.testing.assert_allclose(tie.getTractions(), np.tile(-PENALTY * np.array(d), (tie.nPoints, 1)), atol=1e-9)
        np.testing.assert_allclose(tie.contactForceOnSlave, -PENALTY * SIZE * np.array(d), atol=1e-9)
        np.testing.assert_allclose(tie.contactForceOnSlave + tie.contactForceOnMaster, 0.0, atol=1e-12)


def test_the_tangent_of_a_tie_of_particle_faces_is_the_derivative_of_its_residual_contribution():
    """``K == dP/dU`` by central differences, for tilted faces of two particles astride two elements, one with a gap at
    rest, one with an overlap."""

    model = MPMModel(2)
    master = _masterBlock(model, JournalStandIn())
    particles = [_particle(0.12, 0.03, 1, angle=0.05), _particle(0.40, -0.02, 2, angle=-0.03)]
    tie = _tie(model, particles, master)
    assert (tie.stateOfPoints == ContactState.TIED).all()

    dU = 1e-3 * np.random.default_rng(1).standard_normal(tie.nDof)
    P, K = residualContributionAndTangent(tie, dU.copy(), dU)
    dP_dU = finiteDifferenceTangent(tie, dU.copy(), dU, h=1e-8)
    assert np.abs(P).max() > 0.0
    np.testing.assert_allclose(K, dP_dU, atol=1e-6 * np.abs(K).max())
    np.testing.assert_allclose(K, K.T, atol=1e-12 * np.abs(K).max())


def test_a_tied_particle_carries_its_displacement_from_rest_into_the_next_increment():
    """The particles are updated-Lagrangian: at the next update the tied points are located anew, at the positions the
    particles moved them to, and the tie measures their displacement from rest from there -- it stays tied to the same
    point of the master however far it moves."""

    model = MPMModel(2)
    master = _masterBlock(model, JournalStandIn())
    particle = _particle(0.3, 0.01)
    tie = _tie(model, [particle], master)
    pairing = tie._assignedFacetIdx.copy()

    d = np.array([0.3, 0.4])  # beyond the search distance and the next facet
    _move(particle, d)
    tie.updateConnectivity(model)
    np.testing.assert_array_equal(tie._assignedFacetIdx, pairing)
    _residual(tie)
    np.testing.assert_allclose(tie.contactForceOnSlave, -PENALTY * SIZE * d, rtol=1e-12)
