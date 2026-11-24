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
"""The tie of the nonlocal field of gradient-enhanced particles to finite elements: EdelweissFE's tie
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`, ``tiedFields``, *Tied scalar fields*) with the points
of particle faces (:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`).

The particles are real (Marmot): their nodes carry the increment of the field, and the value of the field at a point of
a particle at the update is taken from the particle's state, its value and gradient at the center. The interface patch
test: a linear field crosses the tie without a force, whether it is in the state of the particles or in the increment
at their nodes, and a uniform jump gives equal and opposite consistent fluxes. Then the tangent, and the critical time
step of a first-order field with the tie's rows against the exact eigenvalue. Skipped where Marmot is not built.
"""

import numpy as np
import pytest
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.generators.planerectquad import Generator as PlaneRectQuadGenerator
from edelweissfe.generators.planerectquad import PlaneRectQuadSchema
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.utils.residualtangentcheck import (
    finiteDifferenceTangent,
    residualContributionAndTangent,
)

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
from edelweissmeshfree.models.test._criticaltimestep import serially
from edelweissmeshfree.models.test._particleplate import RHO, H, particlePlate

PARTICLE = "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad"
#: The penalty of the displacement per unit area, the nonlocal radius of the particles' material, the factor of the
#: penalty of the field.
PENALTY, RADIUS, FACTOR = 1e4, 5.0, 10.0
#: A linear field, small enough to leave the material elastic: its value at the origin and its gradient.
A, B = 2e-5, np.array([3e-7, -2e-7])


def _tiedPlate(nQuadraturePoints: int = 2):
    """The plate of 8 x 4 gradient-enhanced particles of the side H, its bottom faces tied to a block of 4 plane strain
    elements below it, the displacement and the nonlocal field; the model and the tie, paired."""

    model = particlePlate(PARTICLE, 2.2, 3)
    journal = Journal()
    with model.topology.changes():
        # past the labels of the nodes of the kernel functions, which their generator takes as they are
        model.topology.reserveNodeNumbers(max(model.nodes))
        PlaneRectQuadGenerator(
            "block",
            model,
            journal,
            configuration=PlaneRectQuadSchema(x0=0.0, y0=-H, length=8 * H, h=H, nX=4, nY=1, elType="CPE4"),
        )
        facets = model.elementSets[
            buildContactFacets(model, "block_top", "top", "midside", "facetConsistent", journal)[0]
        ]
    bottomRow = [p for p in model.particles.values() if p.getCenterCoordinates()[1] < H]
    tie = SurfaceContact(
        "tie",
        model,
        ContactPointsOnParticleFaces(
            [(particle, 1) for particle in bottomRow], nQuadraturePoints, interpolationLinearizedAboutCenter=True
        ),
        ElementSet("master", list(facets)),
        journal,
        configuration=SurfaceToDeformableSurfacePenaltySchema(
            penalty=PENALTY,
            contactType="tied",
            tiedFields="displacement; nonlocal damage",
            nonlocalRadius=RADIUS,
            nonlocalPenaltyFactor=FACTOR,
        ),
    )
    tie.updateConnectivity(model)
    return model, tie


def _linear(x: np.ndarray) -> np.ndarray:
    return A + x @ B


def _fieldOfTheTie(tie) -> tuple:
    """The degrees of freedom of the nonlocal field in the block of the tie: at the slave nodes, at the master."""

    field = tie._fieldsInTheBlock[1]
    return field.slave, field.master


def _addALinearIncrementOfTheFieldToTheParticles(model):
    """Every particle takes an increment that is the linear field at the nodes of its kernel functions, with no
    displacement, and accepts it: its state is then the linear field, at its center and in its gradient."""

    for particle in model.particles.values():
        nodes = [kernel.node for kernel in particle.kernelFunctions]
        dU = np.zeros((len(nodes), 3))
        dU[:, 2] = _linear(np.array([node.coordinates for node in nodes]))
        nDof = particle.nDof
        particle.computeKernels(np.zeros(nDof * nDof), np.zeros(nDof), dU.ravel(), dU.ravel(), 1.0, 1.0)
        particle.acceptStateAndPosition()


def _residualOfTheField(tie, U: np.ndarray, dU: np.ndarray) -> np.ndarray:
    P = np.zeros(tie.nDof)
    tie.computeKernels(U, dU, P, None, None)
    return P[tie._fieldsInTheBlock[1].slave.start :]


@pytest.mark.parametrize("nQuadraturePoints", [1, 2])
def test_a_linear_field_crosses_the_tie_from_the_increment_at_the_nodes_and_from_the_state(nQuadraturePoints):
    """The linear field on the elements (their total) and on the particles: first as the increment at the nodes of the
    kernel functions, which the particles' interpolation reproduces at the points; then accepted into the particles'
    state, from which the tie takes the values at the points at the next update. No jump, no force, either way; and
    the values from the state are what the increment added up to."""

    model, tie = _tiedPlate(nQuadraturePoints)
    slave, master = _fieldOfTheTie(tie)
    X = np.array([node.coordinates for node in tie._nodes])
    np.testing.assert_array_equal(tie._valuesAtTheOriginsOfLocated[0], 0.0)

    U, dU = np.zeros(tie.nDof), np.zeros(tie.nDof)
    dU[slave] = U[slave] = _linear(X[: tie._nSlaveNodes])
    U[master] = _linear(X[tie._nSlaveNodes :])
    # the round-off of the forces of the springs, the stiffness of the stiffest times the largest value of the field
    roundOff = 1e-13 * np.max(tie._scalarPenaltyTimesAreaOfLocated) * np.max(np.abs(U[master]))
    np.testing.assert_allclose(_residualOfTheField(tie, U, dU), 0.0, atol=roundOff)
    increment = tie.slave.changesFromTheOriginsOfLocatedPoints(U[slave], dU[slave], 1)[:, 0]

    _addALinearIncrementOfTheFieldToTheParticles(model)
    tie.updateConnectivity(model)
    points = tie.slave._coordinatesOfAllPoints()[tie._locatedPoints]
    np.testing.assert_allclose(tie._valuesAtTheOriginsOfLocated[0], _linear(points), rtol=1e-10)
    np.testing.assert_allclose(tie._valuesAtTheOriginsOfLocated[0], increment, rtol=1e-10)

    dU[:] = 0.0
    U[slave] = 0.0
    np.testing.assert_allclose(_residualOfTheField(tie, U, dU), 0.0, atol=roundOff)


def test_a_uniform_jump_gives_equal_and_opposite_fluxes():
    """The master below the particles by a uniform jump j: the particles are pulled down and the master up by the
    flux kappa j over the tied faces, kappa = beta l^2 / h with the size h of the elements' faces."""

    model, tie = _tiedPlate()
    slave, master = _fieldOfTheTie(tie)
    jump = 1e-4
    U = np.zeros(tie.nDof)
    U[master] = -jump
    P = np.zeros(tie.nDof)
    tie.computeKernels(U, np.zeros_like(U), P, None, None)

    kappa = FACTOR * RADIUS**2 / (2 * H)
    flux = kappa * jump * 8 * H
    np.testing.assert_allclose(P[slave].sum(), flux, rtol=1e-12)
    np.testing.assert_allclose(P[master].sum(), -flux, rtol=1e-12)


def test_the_tangent_is_the_derivative_of_the_residual_contribution():
    model, tie = _tiedPlate()
    rng = np.random.default_rng(4)
    dU = 1e-3 * rng.standard_normal(tie.nDof)
    U = dU + rng.standard_normal(tie.nDof)
    P, K = residualContributionAndTangent(tie, U, dU)
    np.testing.assert_allclose(K, finiteDifferenceTangent(tie, U, dU, h=1e-8), atol=1e-6 * np.abs(K).max())


@pytest.mark.parametrize("order", ["first", "second"])
def test_the_rows_of_the_tied_field_bound_the_exact_eigenvalue(order):
    """The nonlocal field of the particles, tied to held elements, of the first order (a damping, no inertia: forward
    Euler, :math:`\\Delta t \\le 2/\\lambda_\\mathrm{max}(\\boldsymbol{C}^{-1}\\boldsymbol{K})`) or of the second (a
    micro-inertia: central differences, :math:`\\Delta t \\le 2/\\sqrt{\\lambda_\\mathrm{max}(\\boldsymbol{M}^{-1}
    \\boldsymbol{K})}`): the rows of the tie, scaled by the lumped coefficient of the field on both sides, added to the
    particles' field block, give a critical time step below the exact one of the particles' nonlocal block with the
    tie's springs; and with the springs the exact limit is lower than without them."""

    from edelweissmeshfree.models.mpmdiscretesystem import (
        STABLE_FRACTION_OF_THE_TANGENT_LIMIT,
    )

    model, tie = _tiedPlate()
    system = model.buildDiscreteSystem()
    nDof = system.numbering.nDof
    indexOf = system.numbering.idcsOfFieldVariablesInDofVector

    # an inertia far above the particles' on the displacement, so that the field sets the critical time step; on the
    # field, the damping of a nonlocal viscosity eta or the micro-inertia eta^2/4, lumped as the mass
    M = np.asarray(system.assembleLumpedMassOfEntitiesWithOwnInertia(serially)).copy()
    eta = 1e-4
    coefficient = np.zeros(nDof)
    for kernel in model.meshfreeKernelFunctions.values():
        node = kernel.node
        lumpedVolume = M[indexOf[node.fields["displacement"]][0]] / RHO
        coefficient[indexOf[node.fields["nonlocal damage"]][0]] = lumpedVolume * (
            eta if order == "first" else eta**2 / 4
        )
    ofTheField = coefficient > 0.0
    M[ofTheField] = 0.0
    M *= 1e12
    C = np.zeros(nDof)
    if order == "first":
        C[ofTheField] = coefficient[ofTheField]
    else:
        M[ofTheField] = coefficient[ofTheField]
    s = 1.0 / np.sqrt(coefficient[ofTheField])

    K = np.zeros((nDof, nDof))
    for particle in system.particles:
        stiffness = np.zeros(particle.nDof**2)
        particle.assembleStiffness(stiffness)
        indices = np.asarray(system.locations.ofEntity[particle])
        K[np.ix_(indices, indices)] += stiffness.reshape(particle.nDof, particle.nDof, order="F")
    ofTheParticles = K.copy()

    # the tie's springs of the field, on the particles' nodes (the elements held: zero weight, no row)
    slave, _ = _fieldOfTheTie(tie)
    slaveIndices = np.array([indexOf[node.fields["nonlocal damage"]][0] for node in tie._nodes[: tie._nSlaveNodes]])
    Ktie = residualContributionAndTangent(tie, np.zeros(tie.nDof), np.zeros(tie.nDof))[1][slave, slave]
    K[np.ix_(slaveIndices, slaveIndices)] += Ktie

    def exact(stiffness):
        S = s[:, None] * stiffness[np.ix_(ofTheField, ofTheField)] * s[None, :]
        rho = np.max(np.abs(np.linalg.eigvals(S)))
        return 2.0 / rho if order == "first" else 2.0 / np.sqrt(rho)

    weights = np.zeros(tie.nDof)
    scaling = np.zeros(nDof)
    scaling[ofTheField] = s
    weights[slave] = scaling[slaveIndices]
    bound = np.zeros(tie.nDof)
    tie.addBoundOfAbsoluteStiffnessTimes(np.zeros(tie.nDof), np.zeros(tie.nDof), weights, bound, None)
    rowSums = np.zeros(nDof)
    np.add.at(rowSums, slaveIndices, bound[slave])
    rowSums *= scaling

    estimate = system.criticalTimeStepOfEntitiesWithOwnInertia(M, C, serially)
    withTheTie = estimate.criticalTimeStep(rowSums) / STABLE_FRACTION_OF_THE_TANGENT_LIMIT
    alone = estimate.value / STABLE_FRACTION_OF_THE_TANGENT_LIMIT

    assert exact(K) < exact(ofTheParticles)
    assert alone <= exact(ofTheParticles)
    assert withTheTie <= exact(K)
    assert withTheTie > 0.25 * exact(K)
