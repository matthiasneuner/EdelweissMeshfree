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
"""An explicit axisymmetric impact: a ring punch of finite elements (``CX8RUL``, steel) hits a ring of axisymmetric
gradient-enhanced particles (``GradientEnhancedFiniteStrainSQCNIxNSNI/Axisymmetric/Quad``, the particle of the
axisymmetric screw anchor), the two meeting through EdelweissFE's penalty contact on the faces of the particles,
integrated by ``NEDParallel``. The particles' material is the compressible neo-Hookean solid with nonlocal damage,
whose threshold lies far above the strains of the impact: the block stays elastic. (The displacement particle
``DisplacementSQCNI/Axisymmetric/Quad`` has no explicit path.)

.. code-block:: text

      z
      |            +-------+        the punch (FE), moving down at V0
      |            | punch |
      |            +-------+
      |  +======================+   contact: the top faces of the particles against the punch's bottom
      |  |   block of particles |
      |  |        (free)        |
      |  +----------------------+
      +---------------------------- r
         R0                R0 + WIDTH

The model is the section r-z of a body of revolution: the punch and the block are rings about the axis, every contact
point stands for its ring, and the block is free -- nothing but the contact acts on the model. So the checks are those
of a conservative system:

- **Momentum**: the contact forces on the two sides cancel in every increment, and the axial momentum of the model
  stays the punch's initial one, up to the change of the masses the particles lump to their nodes, whose velocities
  are carried from increment to increment (a few parts in a hundred thousand, as in plane strain).
- **Energy**: the kinetic energy, the strain energy of the punch (which its elements publish) and of the particles (the
  neo-Hookean energy of their deformation gradient on the volume of their rings) and the energy stored in the springs
  of the contact add up to the punch's initial kinetic energy, to within the error of the central differences: the
  penalty contact gains a little energy as it closes and loses some as it opens, within an increment, a few percent at
  the courant number 0.8, shrinking with the time increment. The plane strain twin of this model loses the same. The
  punch is slowed down nearly to rest, and the block set in motion.
- **The critical time step**: the increments ``NED`` takes, and its estimate of the critical time step of the
  particles, stay below the exact critical time step of the block's particles (from the eigenvalues of their assembled
  stiffness and lumped mass at the start). Here the particles set the time increment. With
  ``carry-critical-time-step-weights`` the run is the same: the contact acts on the displacement field of the
  particles, whose weights then stay constant per field (see
  :meth:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`), so
  that the rows of its springs add to the particles' rows as they are; the carried weights tighten the estimate of the
  particles where no penalty constraint acts on their field
  (``edelweissmeshfree/models/test/criticaltimestepofparticles_test.py``).

Units: N, mm, t, s.
"""

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
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
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
)
from edelweissmeshfree.generators.rectangularquadparticlegridgenerator import (
    generateRectangularQuadParticleGrid,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test._criticaltimestep import exactCriticalTimeStep
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

#: The block: inner radius, width, height and the size of its particles [mm].
R0, WIDTH, HEIGHT, PARTICLE = 10.0, 20.0, 20.0, 2.5
#: The punch: its inner radius, width and height [mm], the gap to the block at the start, and its velocity [mm/s].
PUNCH_R0, PUNCH_WIDTH, PUNCH_HEIGHT, GAP, V0 = 15.0, 10.0, 10.0, 0.01, -5e3


def neoHooke(youngsModulus: float, poissonsRatio: float, density: float) -> np.ndarray:
    """The properties of a compressible neo-Hookean solid: bulk modulus, shear modulus, density."""

    return np.array([youngsModulus / (3 * (1 - 2 * poissonsRatio)), youngsModulus / (2 * (1 + poissonsRatio)), density])


CONCRETE, STEEL = neoHooke(30000.0, 0.2, 2.4e-9), neoHooke(210000.0, 0.3, 7.85e-9)
#: GRADIENTENHANCEDCOMPRESSIBLENEOHOOKEDAMAGE: bulk and shear modulus, the damage threshold and final value of the
#: nonlocal driving force (far above the impact's), the nonlocal length [mm], density, and the nonlocal viscosity [s],
#: with which NED integrates the nonlocal field in time (large enough not to limit the time increment).
CONCRETE_WITH_DAMAGE = np.array([CONCRETE[0], CONCRETE[1], 1.0, 2.0, 2.0, CONCRETE[2], 1e-4])
#: The penalty of the contact per unit area [N/mm^3], about the stiffness of a particle per its size, and its search
#: distance.
PENALTY, SEARCH_DISTANCE = 1e4, 1.0
#: The duration of the run: the punch meets the block after GAP / |V0|, 2 us, and presses it for a few transits of the
#: compression wave through the block.
END_TIME = 20e-6


def neoHookeanEnergyDensity(F: np.ndarray, properties: np.ndarray) -> float:
    """The energy density of Marmot's compressible neo-Hookean solid (Pence and Gou, potential B), undamaged, at the
    deformation gradient ``F``: :math:`\\frac{K}{8}(J^2 + J^{-2} - 2) + \\frac{G}{2}(\\mathrm{tr}\\,\\boldsymbol{C}\\, J^{-2/3} - 3)`.
    """

    bulkModulus, shearModulus = properties[:2]
    J2 = np.linalg.det(F) ** 2
    return bulkModulus / 8 * (J2 + 1 / J2 - 2) + shearModulus / 2 * (np.sum(F * F) * J2 ** (-1 / 3) - 3)


def buildModel(journal: Journal) -> tuple[MPMModel, SurfaceContact]:
    """The punch, the block and their contact, prepared."""

    model = MPMModel(2, axisymmetric=True)
    with model.topology.changes():
        PlaneRectQuadGenerator(
            "punch",
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=PUNCH_R0, y0=GAP, length=PUNCH_WIDTH, h=PUNCH_HEIGHT, nX=1, nY=1, elType="CX8RUL"
            ),
        )
    model.sections["punch"] = PlaneSection(
        "punch",
        model,
        {"name": "CompressibleNeoHooke", "properties": STEEL},
        [model.elementSets["punch_all"]],
        configuration=PlaneSectionSchema(thickness=1.0),
    )
    with model.topology.changes():
        punchBottom = buildContactFacets(model, "punch_bottom", "punch_bottom", "midside", "facetConsistent", journal)[
            0
        ]

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    material = {"material": "GRADIENTENHANCEDCOMPRESSIBLENEOHOOKEDAMAGE", "properties": CONCRETE_WITH_DAMAGE}
    generateRectangularQuadParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "GradientEnhancedFiniteStrainSQCNIxNSNI/Axisymmetric/Quad", number, vertices, 0.0, approximation, material
        ),
        name="block",
        x0=R0,
        y0=-HEIGHT,
        l=WIDTH,
        h=HEIGHT,
        nX=round(WIDTH / PARTICLE),
        nY=round(HEIGHT / PARTICLE),
    )
    generateKernelMatchingToParticle(
        model,
        journal,
        lambda node, supportSize: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3
        ),
        model.particleSets["block_all"],
        supportScalingFactor=2.4,
        firstKernelFunctionNumber=max(model.nodes) + 1,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["block"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    )

    contact = SurfaceContact(
        "contact",
        model,
        ContactPointsOnParticleFaces(
            model.surfaces["block_top"], nQuadraturePoints=2, interpolationLinearizedAboutCenter=True
        ),
        model.elementSets[punchBottom],
        journal,
        configuration=SurfaceToDeformableSurfacePenaltySchema(penalty=PENALTY, searchDistance=SEARCH_DISTANCE),
    )
    model.constraints["contact"] = contact
    model.prepareYourself(journal)
    return model, contact


def nodesOfThePunch(model: MPMModel) -> list:
    """The nodes of the punch's elements, each once."""

    return list(dict.fromkeys(node for element in model.elementSets["punch_all"] for node in element.nodes))


class ImpactRecord(OutputManagerBase):
    """Records, in every increment, what the checks need: the time, the axial momentum, the energies, the contact
    forces on both sides, the punch's velocity, and the particles' estimate of their critical time step."""

    checkpointedState = {}

    def __init__(self, model: MPMModel, solver: NEDParallel, contact: SurfaceContact):
        self.name = "impactRecord"
        self._model, self._solver, self._contact = model, solver, contact
        displacement = model.nodeFields["displacement"]
        self._punchNodes = [displacement.indexOfNode(node) for node in nodesOfThePunch(model)]
        self.rows = []

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        model, solver, contact = self._model, self._solver, self._contact
        strainEnergyOfTheParticles = sum(
            neoHookeanEnergyDensity(particle.getResultArray("deformation gradient").reshape(3, 3), CONCRETE)
            * particle.getVolumeUndeformed()
            for particle in model.particleSets["block_all"]
        )
        strainEnergyOfThePunch = sum(element.computeInternalEnergy() for element in model.elementSets["punch_all"])
        self.rows.append(
            dict(
                time=model.time,
                axialMomentum=float(solver.secondOrderMomentum(solver._rawLumpedMass, solver._V, model)[1]),
                kineticEnergy=solver.kineticEnergy(solver._V),
                strainEnergy=strainEnergyOfTheParticles + strainEnergyOfThePunch,
                penaltyEnergy=contact.penaltyEnergy,
                actionPlusReaction=contact.contactForceOnSlave + contact.contactForceOnMaster,
                forceOnTheBlock=float(contact.contactForceOnSlave[1]),
                velocityOfThePunch=float(np.mean(model.nodeFields["displacement"]["V"][self._punchNodes, 1])),
                criticalTimeStepOfTheParticles=solver._criticalTimeStepOfTheEntities.value,
            )
        )

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


def run(carryWeights: bool = False, courantNumber: float = 0.8) -> tuple[list, float, float]:
    """Run the impact.

    Parameters
    ----------
    carryWeights
        ``NED``'s ``carry-critical-time-step-weights``.
    courantNumber
        ``NED``'s ``courant-number``.

    Returns
    -------
    tuple[list, float, float]
        The recorded rows, the punch's initial kinetic energy and axial momentum.
    """

    journal = Journal(verbose=False)
    model, contact = buildModel(journal)

    displacement = model.nodeFields["displacement"]
    for node in nodesOfThePunch(model):
        displacement["V"][displacement.indexOfNode(node), 1] = V0

    jobInfo = loadConfiguration({})
    solver = NEDParallel(
        jobInfo,
        journal,
        **{"courant-number": courantNumber, "output-frequency": 1, "carry-critical-time-step-weights": carryWeights},
    )
    record = ImpactRecord(model, solver, contact)
    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    step = AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [record],
        StepActionCollection(),
        stepLength=END_TIME,
        startInc=1.0,
        maxInc=1.0,
        minInc=1e-12,
        maxNumInc=10**5,
    )
    step.solve()
    fieldOutputController.finalizeJob()

    punchMass = STEEL[2] * np.pi * ((PUNCH_R0 + PUNCH_WIDTH) ** 2 - PUNCH_R0**2) * PUNCH_HEIGHT
    return record.rows, 0.5 * punchMass * V0**2, punchMass * V0


def exactCriticalTimeStepOfTheBlock() -> float:
    """The critical time step of the block's particles at the start, from the eigenvalues of their assembled system."""

    model, _ = buildModel(Journal(verbose=False))
    return exactCriticalTimeStep(model)


def largestEnergyError(rows: list, initialKineticEnergy: float) -> float:
    """The largest deviation of the total energy from the punch's initial kinetic energy, relative to it."""

    totals = np.array([row["kineticEnergy"] + row["strainEnergy"] + row["penaltyEnergy"] for row in rows])
    return float(np.abs(totals / initialKineticEnergy - 1.0).max())


def checkConservation(rows: list, initialKineticEnergy: float, initialMomentum: float):
    """Momentum in every increment, energy to within the error of the central differences, and the impact happened."""

    largestForce = max(abs(row["forceOnTheBlock"]) for row in rows)
    assert largestForce > 0.0
    # the lumped mass of the punch's rings is the mass of the ring
    assert rows[0]["axialMomentum"] == pytest.approx(initialMomentum, rel=1e-6)
    # the nodes of the particles carry their velocity from increment to increment, and the particles lump their mass
    # to them anew in every increment: the momentum changes by the velocity times the change of the lumped masses, a
    # few parts in a hundred thousand here, as in the plane strain twin of the model
    for row in rows:
        assert row["axialMomentum"] == pytest.approx(rows[0]["axialMomentum"], rel=1e-4)
        np.testing.assert_allclose(row["actionPlusReaction"], 0.0, atol=1e-9 * largestForce)
    assert largestEnergyError(rows, initialKineticEnergy) < 0.06
    # the punch gave nearly all of its energy to the block
    assert rows[-1]["velocityOfThePunch"] > 0.1 * V0


def test_the_impact_conserves_momentum_and_energy_and_stays_below_the_critical_time_step():
    rows, initialKineticEnergy, initialMomentum = run()
    checkConservation(rows, initialKineticEnergy, initialMomentum)

    exact = exactCriticalTimeStepOfTheBlock()
    increments = np.diff([row["time"] for row in rows])
    assert increments.max() <= exact
    assert max(row["criticalTimeStepOfTheParticles"] for row in rows) <= exact


def test_the_energy_error_of_the_contact_shrinks_with_the_time_increment():
    rows, initialKineticEnergy, _ = run()
    halved, _, _ = run(courantNumber=0.4)

    assert largestEnergyError(halved, initialKineticEnergy) < 0.7 * largestEnergyError(rows, initialKineticEnergy)


def test_with_carried_weights_and_contact_on_the_particles_the_run_is_the_same():
    rows, initialKineticEnergy, initialMomentum = run()
    carried, _, _ = run(carryWeights=True)
    checkConservation(carried, initialKineticEnergy, initialMomentum)

    assert len(carried) == len(rows)
    np.testing.assert_allclose(
        [row["criticalTimeStepOfTheParticles"] for row in carried],
        [row["criticalTimeStepOfTheParticles"] for row in rows],
        rtol=1e-6,
    )


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


if __name__ == "__main__":
    exact = exactCriticalTimeStepOfTheBlock()
    for carryWeights in (False, True):
        rows, initialKineticEnergy, initialMomentum = run(carryWeights)
        totals = np.array([row["kineticEnergy"] + row["strainEnergy"] + row["penaltyEnergy"] for row in rows])
        estimates = np.array([row["criticalTimeStepOfTheParticles"] for row in rows])
        print(
            f"carried weights {carryWeights}: {len(rows)} increments, energy {totals.min() / initialKineticEnergy:.6f}"
            f" to {totals.max() / initialKineticEnergy:.6f} of the initial, the particles' estimate"
            f" {estimates.min():.4e} to {estimates.max():.4e} s, exact {exact:.4e} s"
            f" (ratio {exact / estimates.max():.3f}),"
            f" punch velocity at the end {rows[-1]['velocityOfThePunch']:.4e} mm/s"
        )
