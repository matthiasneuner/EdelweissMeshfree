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
Implicit (quasi-static) frictional discrete rigid body contact test.

The rigid body reference point is kinematically indented into the block and
then dragged sideways via Dirichlet boundary conditions, solved with the
Newton-based :class:`NonlinearQuasistaticSolver` and the implicit
:class:`FrictionalDiscreteRigidBodyPenaltyContact` constraint. Verifies that
the resulting tangential-to-normal force ratio approaches the prescribed
friction coefficient.

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
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.constraints.frictionaldiscreterigidbodypenaltycontact import (
    FrictionalDiscreteRigidBodyPenaltyContact,
)
from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
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
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmeshfree.stepactions.dirichlet import Dirichlet as DirichletMF

E = 2.0
NU = 0.3
RHO = 1.0e-9
PENALTY = 1.0e5
MU = 0.3
DISP_Y = -0.7
DISP_X = 0.1
BLOCK_SIZE = 1.0


def run_sim(full: bool = False):
    # `full=True` restores the original, full-size scenario (10x10x10 block,
    # 25 increments) for visual inspection; the pytest default is a tiny,
    # fast smoke test.
    N = 10 if full else 2  # particles per side of the (N x N x N) block
    STEP_TIME = 0.25 if full else 0.05
    DT = 0.01  # 25 (full) or 5 (fast) increments

    theJournal = Journal()
    theModel = MPMModel(3)

    app = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)

    def theParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/3D/Hexa",
            number,
            vertexCoordinates,
            0.0,
            app,
            {"material": "CompressibleNeoHooke", "properties": np.array([E, NU, RHO]), "plane state": "none"},
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
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, 3, theJournal, bondParticlesToKernelFunctions=True
    )

    rigid_body = generateDiscreteRigidBodyFromMeshFile(
        theModel,
        theJournal,
        name="rigid_body",
        filename="rigid_body.exo",
        translation=np.array([0.0, -9.9, 0.0]),
        density=RHO,
    )

    def amplitude(t):
        return t

    rp_bc_y = DirichletMF(
        "rp_bc_y", theModel.nodeSets["rigid_body_rp"], "displacement", {"2": DISP_Y}, theModel, theJournal, amplitude
    )
    rp_bc_x = DirichletMF(
        "rp_bc_x", theModel.nodeSets["rigid_body_rp"], "displacement", {"1": DISP_X}, theModel, theJournal, amplitude
    )
    rp_bc_z = DirichletMF(
        "rp_bc_z", theModel.nodeSets["rigid_body_rp"], "displacement", {"3": 0.0}, theModel, theJournal
    )
    rp_bc_rot = DirichletMF(
        "rp_bc_rot",
        theModel.nodeSets["rigid_body_rp"],
        "rotation",
        {"1": 0.0, "2": 0.0, "3": 0.0},
        theModel,
        theJournal,
    )

    # Fix the bottom face vertices of the block against the ground.
    for p in theModel.particles.values():
        bottom_verts = [
            v_idx for v_idx, v_coord in enumerate(p.getVertexCoordinates()) if v_coord[1] < -BLOCK_SIZE * N + 0.1
        ]
        if bottom_verts:
            constraint = ParticlePenaltyWeakDirichlet(
                f"bc_bottom_{p.number}",
                theModel,
                [p],
                "displacement",
                {0: 0.0, 1: 0.0, 2: 0.0},
                1e8,
                constrain=bottom_verts,
            )
            theModel.constraints[constraint.name] = constraint

    contact_constraint = FrictionalDiscreteRigidBodyPenaltyContact(
        name="contact",
        particles=theModel.particles.values(),
        model=theModel,
        rigidBody=rigid_body,
        frictionCoefficient=MU,
        viscousRegularization=1.0e6,
        penaltyParameter=PENALTY,
        proximityFactor=2.0,
    )
    theModel.constraints["contact"] = contact_constraint

    theModel.prepareYourself(theJournal)

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

    solver = NonlinearQuasistaticSolver(theJournal)
    incSize = DT / STEP_TIME
    timeStepper = AdaptiveTimeStepper(0.0, STEP_TIME, incSize, incSize, incSize, 50, theJournal)

    solver.solveStep(
        timeStepper,
        pardisoSolve,
        theModel,
        fieldOutputController,
        outputManagers=[ensightOutput],
        particleManagers=[theParticleManager],
        dirichlets=[rp_bc_y, rp_bc_x, rp_bc_z, rp_bc_rot],
        constraints=list(theModel.constraints.values()),
        userIterationOptions={
            "max. iterations": 20,
            "critical iterations": 10,
            "allowed residual growths": 5,
            "default absolute flux residual tolerance": 1.0e-3,
            "default absolute field correction tolerance": 1.0e-5,
        },
    )

    fieldOutputController.finalizeJob()
    ensightOutput.finalizeJob()

    return fieldOutputController, contact_constraint


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    fieldOutputController, contact_constraint = run_sim()

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()
    gold = np.loadtxt("gold_implicit_friction_block.csv")
    assert_gold(res, gold, atol=1e-8)

    # At this drastically shrunk scale and increment count, the sliding
    # contact hasn't necessarily reached the Coulomb slip limit yet, so we
    # only check that both a normal and a bounded tangential reaction force
    # developed, not that their ratio matches MU exactly.
    fn = abs(contact_constraint.totalNormalForce[1])
    ft = abs(contact_constraint.totalFrictionForce[0])
    assert fn > 0.0
    assert 0.0 < ft <= MU * fn * 1.05


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    parser.add_argument(
        "--full", dest="full", action="store_true", help="run the full-size scenario instead of the fast smoke test."
    )
    args = parser.parse_args()

    fieldOutputController, contact_constraint = run_sim(full=args.full)
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    fn = abs(contact_constraint.totalNormalForce[1])
    ft = abs(contact_constraint.totalFrictionForce[0])
    print(f"Final normal force (Y): {fn:.4e} N")
    print(f"Final friction force (X): {ft:.4e} N")
    if fn > 0:
        print(f"Ratio Ft / Fn: {ft / fn:.4f} (expected ~{MU})")

    if args.create_gold:
        if args.full:
            raise SystemExit("Refusing to overwrite the gold file with a --full run; run without --full.")
        np.savetxt("gold_implicit_friction_block.csv", np.asarray(res).flatten())
