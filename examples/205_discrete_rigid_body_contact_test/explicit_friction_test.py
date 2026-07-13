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
"""
Frictional explicit discrete rigid body contact test.

The rigid body falls onto the block with an additional tangential (sliding)
velocity component, exercising the Coulomb-like friction model in
:class:`FrictionalDiscreteRigidBodyPenaltyContactExplicit`. Verifies that a
non-negligible tangential (frictional) reaction force develops once contact
is established.

The pytest-collected :func:`test_sim` uses a tiny block and few increments
to stay fast; pass ``--full`` on the command line for a full-size,
many-increment run suitable for visual inspection; Ensight output is always
written.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.generators.discreterigidbodygenerator import (
    generateDiscreteRigidBodyFromMeshFile,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.constraints.explicit.frictionaldiscreterigidbodypenaltycontactexplicit import (
    FrictionalDiscreteRigidBodyPenaltyContactExplicitFactory,
)
from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticleExplicitPenaltyCartesianBoundaryConstraintFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.boxhexaparticlegridgenerator import (
    generateBoxHexaParticleGrid,
)
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.verletlistparticlemanager import (
    VerletListParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.explicitmultiphysicssolver import (
    ExplicitMultiphysicsSolver,
)

E = 5.0e6
NU = 0.45
RHO = 1000.0
PENALTY = 1e7
VELOCITY = 60.0
SLIDE_VELOCITY = 20.0
FRICTION_COEFFICIENT = 0.5
BLOCK_SIZE = 1.0


def run_sim(full: bool = False):
    # `full=True` restores the original, full-size scenario (10x10x10 block,
    # ~100 increments) for visual inspection; the pytest default is a tiny,
    # fast smoke test.
    N = 10 if full else 2  # particles per side of the (N x N x N) block
    STEP_TIME = 0.03 if full else 0.02
    DT = 1e-3 if full else 2e-4

    theJournal = Journal()
    theModel = MPMModel(3)

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)
    theMaterial = {"material": "CompressibleNeoHooke", "properties": np.array([E, NU, RHO])}

    def theParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/3D/Hexa", number, vertexCoordinates, 0.0, theApproximation, theMaterial
        )

    theModel = generateBoxHexaParticleGrid(
        theModel,
        theJournal,
        theParticleFactory,
        name="block",
        x0=-BLOCK_SIZE * N / 2,
        y0=-BLOCK_SIZE * N,
        z0=-BLOCK_SIZE * N / 2,
        l=BLOCK_SIZE * N,
        h=BLOCK_SIZE * N,
        t=BLOCK_SIZE * N,
        nX=N,
        nY=N,
        nZ=N,
    )

    def theMeshfreeKernelFunctionFactory(node, characteristicLength):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=characteristicLength, continuityOrder=3
        )

    theModel = generateKernelMatchingToParticle(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        theModel.particleSets["block_all"],
        supportScalingFactor=2.2,
    )

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theModel.particleKernelDomains["all"] = theParticleKernelDomain
    theParticleManager = VerletListParticleManager(
        theParticleKernelDomain, 3, theJournal, bondParticlesToKernelFunctions=True
    )

    bottom_particles = [
        p for p in theModel.particleSets["block_all"] if p.getCenterCoordinates()[1] < -BLOCK_SIZE * N + 0.5
    ]
    theModel.particleSets["bottom"] = bottom_particles

    dirichletBottom = ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
        "bottom_fix",
        boundaryPosition=-BLOCK_SIZE * N,
        component=1,
        particleCollection=theModel.particleSets["bottom"],
        field="displacement",
        model=theModel,
        location="center",
        penaltyParameter=PENALTY,
    )
    theModel.constraints.update(dirichletBottom)

    rigid_body = generateDiscreteRigidBodyFromMeshFile(
        theModel,
        theJournal,
        name="rigid_body",
        filename="rigid_body.exo",
        translation=[0.0, -9.6, 0.0],
        density=RHO,
        initial_velocity=[0.0, -VELOCITY, SLIDE_VELOCITY],
    )

    contact = FrictionalDiscreteRigidBodyPenaltyContactExplicitFactory(
        name="rigid_impact",
        particleCollection=theModel.particleSets["block_all"],
        model=theModel,
        rigidBody=rigid_body,
        penaltyParameter=PENALTY,
        frictionCoefficient=FRICTION_COEFFICIENT,
        viscousRegularization=1e5,
    )

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput("displacement", theModel.particleSets["block_all"], "displacement")
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements", theModel.particleSets["block_all"], "vertex displacements", reshape_to_dimensions=3
    )
    fieldOutputController.addPerNodeFieldOutput(
        "rigid_displacement", theModel.nodeFields["displacement"].subset(rigid_body), "U"
    )
    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        name="vertex displacements",
        create="perNode",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["rigid_displacement"], create="perNode"
    )
    ensightOutput.initializeJob()

    solver = ExplicitMultiphysicsSolver(theJournal)
    incSize = DT / STEP_TIME
    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, STEP_TIME, incSize, incSize, incSize, 4000, theJournal)

    solver.solveStep(
        adaptiveTimeStepper,
        theModel,
        fieldOutputController,
        outputManagers=[ensightOutput],
        particleManagers=[theParticleManager],
        userIterationOptions={"field orders": {"displacement": 2, "rotation": 2}},
    )

    fieldOutputController.finalizeJob()
    ensightOutput.finalizeJob()

    return fieldOutputController, contact


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    fieldOutputController, contact = run_sim()

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()
    gold = np.loadtxt("gold_explicit_friction_block.csv")
    assert_gold(res, gold, atol=1e-10)

    # A sliding impact with a non-zero friction coefficient must generate a
    # tangential reaction force.
    assert np.linalg.norm(contact.totalFrictionForce) > 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    parser.add_argument(
        "--full", dest="full", action="store_true", help="run the full-size scenario instead of the fast smoke test."
    )
    args = parser.parse_args()

    fieldOutputController, contact = run_sim(full=args.full)
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    print(f"Total friction force: {contact.totalFrictionForce}")
    print(f"Total normal force: {contact.totalNormalForce}")

    if args.create_gold:
        if args.full:
            raise SystemExit("Refusing to overwrite the gold file with a --full run; run without --full.")
        np.savetxt("gold_explicit_friction_block.csv", np.asarray(res).flatten())
