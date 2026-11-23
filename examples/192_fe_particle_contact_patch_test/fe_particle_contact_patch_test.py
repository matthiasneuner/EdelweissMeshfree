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
"""The contact patch test, with particles and with finite elements: a block between two platens of finite elements is
compressed, and the contact must transmit a uniform pressure across both interfaces.

.. code-block:: text

        |  |  |  |  |  |  |  |    the top of the upper platen moves down
        +----------------------+
        |    upper platen      |
        +======================+  contact
        |                      |
        |   block: particles   |
        |   or finite elements |
        |                      |
        +======================+  contact
        |    lower platen      |
        +----------------------+
        ///////////////////////   the bottom of the lower platen is held

The platens and the block are elastic, the platens finite elements (``CPE8UL``), the block a compressible neo-Hookean
solid, all without lateral contraction (Poisson's ratio zero), the block held only on its left side, in x. Its state
is homogeneous: the pressure is the same at every contact point of both interfaces, and their resultant is the
reaction of the platens. (With lateral contraction the block slides along the platens, its contact points no longer
meet the faces of the platens as they did, and the integrated contact transmits the pressure to within a few parts in
a thousand only: pointwise contact on a non-matching interface, see EdelweissFE's contact theory.) The same contact constraint
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`) serves both blocks; only its contact points differ:
the quadrature points of the faces of the elements, or of the faces of the particles
(:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`). The block
starts with a tiny overlap of the platens, so that the contact holds it from the first iteration on.

The step is quasi-static, solved by EdelweissFE's ``NISTParallel``: the implicit path of the contact, with its tangent.
"""

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
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
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
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
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

#: The block, 40 x 20 mm, of particles or elements of 5 mm; the platens 40 x 10 mm.
WIDTH, HEIGHT, PLATEN, SIZE = 40.0, 20.0, 10.0, 5.0
#: The block's overlap of each platen at the start, and the compression of the step.
OVERLAP, COMPRESSION = 1e-4, 0.05


def neoHooke(youngsModulus: float, poissonsRatio: float) -> dict:
    """A compressible neo-Hookean solid: bulk modulus, shear modulus, density."""

    return {
        "name": "CompressibleNeoHooke",
        "properties": np.array(
            [
                youngsModulus / (3 * (1 - 2 * poissonsRatio)),
                youngsModulus / (2 * (1 + poissonsRatio)),
                1.0,
            ]
        ),
    }


PLATEN_MATERIAL, BLOCK_MATERIAL = neoHooke(210000.0, 0.0), neoHooke(30000.0, 0.0)
#: The interface stiffness per unit area.
PENALTY = 1e6


def addFiniteElementBlock(model: MPMModel, journal: Journal, name: str, y0: float, height: float, material: dict):
    """A block of ``CPE8UL`` elements, ``WIDTH`` wide, from ``y0`` up, with its section."""

    with model.topology.changes():
        PlaneRectQuadGenerator(
            name,
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=0.0, y0=y0, length=WIDTH, h=height, nX=round(WIDTH / SIZE), nY=round(height / SIZE), elType="CPE8UL"
            ),
        )
    model.sections[name] = PlaneSection(
        name, model, material, [model.elementSets[f"{name}_all"]], configuration=PlaneSectionSchema(thickness=1.0)
    )


def facets(model: MPMModel, journal: Journal, surface: str) -> str:
    """The contact facets of a surface of finite elements, ``<surface>_facets``."""

    with model.topology.changes():
        return buildContactFacets(model, surface, surface, "midside", "facetConsistent", journal)[0]


def addParticleBlock(model: MPMModel, journal: Journal, y0: float):
    """The block of displacement particles, on kernel functions bonded to them."""

    grid = dict(x0=0.0, y0=y0, l=WIDTH, h=HEIGHT + 2 * OVERLAP, nX=round(WIDTH / SIZE), nY=round(HEIGHT / SIZE))
    generateRectangularKernelFunctionGrid(
        model,
        journal,
        lambda node: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=2 * SIZE, continuityOrder=3
        ),
        name="block",
        firstKernelFunctionNumber=max(model.nodes) + 1,
        **grid,
    )
    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernelImplicitGradient", 2, completenessOrder=1)
    material = {"material": BLOCK_MATERIAL["name"], "properties": BLOCK_MATERIAL["properties"]}
    generateRectangularQuadParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/PlaneStrain/Quad", number, vertices, 0.0, approximation, material
        ),
        name="block",
        **grid,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["block"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    )


def run(block: str) -> tuple[MPMModel, list]:
    """Compress the block between the platens.

    Parameters
    ----------
    block
        "particles" or "finite elements".

    Returns
    -------
    tuple[MPMModel, list]
        The solved model, and the two contact constraints (lower, upper).
    """

    journal = Journal(verbose=False)
    model = MPMModel(2)
    addFiniteElementBlock(model, journal, "lower", -PLATEN, PLATEN, PLATEN_MATERIAL)
    addFiniteElementBlock(model, journal, "upper", HEIGHT, PLATEN, PLATEN_MATERIAL)
    lowerTop, upperBottom = facets(model, journal, "lower_top"), facets(model, journal, "upper_bottom")

    stepActions = StepActionCollection()
    if block == "particles":
        addParticleBlock(model, journal, -OVERLAP)
        bottom = ContactPointsOnParticleFaces(
            model.surfaces["block_bottom"], nQuadraturePoints=2, interpolationLinearizedAboutCenter=True
        )
        top = ContactPointsOnParticleFaces(
            model.surfaces["block_top"], nQuadraturePoints=2, interpolationLinearizedAboutCenter=True
        )
        # the left vertices of the particles on the left side, held in x
        model.constraints["blockLeft"] = ParticlePenaltyWeakDirichlet(
            "blockLeft", model, list(model.particleSets["block_left"]), "displacement", {0: 0.0}, PENALTY * SIZE, [0, 3]
        )
    else:
        addFiniteElementBlock(model, journal, "block", -OVERLAP, HEIGHT + 2 * OVERLAP, BLOCK_MATERIAL)
        bottom = ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, "block_bottom")], 3, "lower")
        top = ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, "block_top")], 3, "upper")
        stepActions["dirichlet"]["blockLeft"] = Dirichlet(
            "blockLeft", model.nodeSets["block_left"], "displacement", {0: 0.0}, model, journal
        )

    configuration = SurfaceToDeformableSurfacePenaltySchema(penalty=PENALTY, searchDistance=1.0)
    contacts = [
        SurfaceContact(name, model, points, model.elementSets[master], journal, configuration=configuration)
        for name, points, master in (("lower", bottom, lowerTop), ("upper", top, upperBottom))
    ]
    for contact in contacts:
        model.constraints[contact.name] = contact

    model.prepareYourself(journal)

    stepActions["dirichlet"]["held"] = Dirichlet(
        "held", model.nodeSets["lower_bottom"], "displacement", {0: 0.0, 1: 0.0}, model, journal
    )
    stepActions["dirichlet"]["pressed"] = Dirichlet(
        "pressed", model.nodeSets["upper_top"], "displacement", {0: 0.0, 1: -COMPRESSION}, model, journal
    )
    fieldOutputController = MPMFieldOutputController(model, journal)
    jobInfo = loadConfiguration({})
    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        NISTParallel(jobInfo, journal, linsolver="pardiso"),
        [],
        stepActions,
        startInc=0.25,
        maxInc=0.25,
        minInc=1e-3,
        maxNumInc=100,
    )
    fieldOutputController.initializeJob()
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()
    return model, contacts


def reactionOfThePress(model: MPMModel) -> float:
    """The vertical force with which the press pushes the top of the upper platen down, positive."""

    displacement = model.nodeFields["displacement"]
    return -displacement["P"][[displacement.indexOfNode(node) for node in model.nodeSets["upper_top"]]][:, 1].sum()


def pressuresAndPositions(model: MPMModel, contacts: list) -> tuple[np.ndarray, np.ndarray]:
    """The pressure and the x coordinate of every contact point of both interfaces."""

    return (
        np.concatenate([contact.getNormalPressures() for contact in contacts]),
        np.concatenate([contact.slave.currentPointCoordinates(model)[:, 0] for contact in contacts]),
    )


def checkEquilibrium(model: MPMModel, contacts: list) -> float:
    """Each interface transmits the force of the press, and the forces on its two sides cancel; returns the force."""

    reaction = reactionOfThePress(model)
    assert reaction > 0.0
    for contact in contacts:
        assert -contact.totalNormalForce == pytest.approx(reaction, rel=1e-8)
        np.testing.assert_allclose(
            contact.contactForceOnSlave + contact.contactForceOnMaster, 0.0, atol=1e-9 * reaction
        )
    return reaction


def test_finite_elements_transmit_a_uniform_pressure():
    """Matched faces of finite elements: the pressure is uniform at every contact point, up to the slight lateral
    stress of the neo-Hookean solid at finite strain, which the block's held left side takes."""

    model, contacts = run("finite elements")
    reaction = checkEquilibrium(model, contacts)
    pressures, _ = pressuresAndPositions(model, contacts)
    np.testing.assert_allclose(pressures, reaction / WIDTH, rtol=1e-5)


def test_particles_transmit_the_pressure_of_the_finite_element_block():
    """A block of particles transmits the force of the block of finite elements, and away from its free ends the same
    uniform pressure. Within a particle of the free ends the pressure varies by a few percent from point to point: the
    reproducing kernels do not reproduce the homogeneous state exactly at a free corner of the block. That deviation is
    confined to the corners: it does not shrink with the particle size there, while the middle of the block converges
    to the uniform pressure (within a few percent at this size, a hundred times closer at a quarter of it)."""

    reference = reactionOfThePress(run("finite elements")[0])
    model, contacts = run("particles")
    reaction = checkEquilibrium(model, contacts)
    assert reaction == pytest.approx(reference, rel=1e-3)

    pressures, x = pressuresAndPositions(model, contacts)
    middle = np.abs(x - 0.5 * WIDTH) < 0.25 * WIDTH
    np.testing.assert_allclose(pressures[middle], reference / WIDTH, rtol=0.03)
    np.testing.assert_allclose(pressures, reference / WIDTH, rtol=0.1)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


if __name__ == "__main__":
    for block in ("finite elements", "particles"):
        model, contacts = run(block)
        pressures, _ = pressuresAndPositions(model, contacts)
        print(
            f"{block}: force of the press {reactionOfThePress(model):.6e}, pressure {pressures.mean():.6e} "
            f"({pressures.min():.6e} to {pressures.max():.6e})"
        )
