# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import ReachedMaxIncrements, StepFailed

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
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.explicitmultiphysicssolver import (
    ExplicitMultiphysicsSolver,
)
from edelweissmeshfree.stepactions.particleprescribedvelocityexplicit import (
    ParticlePrescribedVelocityExplicit,
)

PENALTY = 1e4


def _npmean(ps):
    import numpy as np

    return float(np.mean([p.getResultArray("displacement")[1] for p in ps]))


def run_sim():
    dimension = 2

    # set nump linewidth to 200:
    np.set_printoptions(linewidth=200)
    # set 2 digits after comma:
    np.set_printoptions(precision=2)
    # and let's print all the array:
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    particleSize = 100.0 / 10
    supportRadius = particleSize * 3.0

    heightProjectile = 200
    lengthProjectile = 100
    nXProjectile = int(lengthProjectile / particleSize)
    nYProjectile = int(heightProjectile / particleSize)
    # place projectile above plate, centered in x direction
    x0Projectile = 0
    y0Projectile = 0

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=3)

    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        x0=x0Projectile,
        y0=y0Projectile,
        h=heightProjectile,
        l=lengthProjectile,
        nX=nXProjectile,
        nY=nYProjectile,
        name="projectile",
    )

    # let's define the type of approximation: We would like to have a reproducing kernel approximation of completeness order 1
    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    tMax = 2.0e-3

    theMaterialProjectile = {
        "material": "GCDP/HUGHES-WINGET",
        # E, nu, fcy, fcu, fbu, ftu, Df, Ah, Bh, Ch, Dh, As, epsF, lDamage, m, maxDamage,
        # derivativeMethod, dTThreshold, viscosity, density, nonlocalViscosity, microInertia
        "properties": np.array(
            [
                30600.0,
                0.2,
                9.3,
                32.3,
                37.0,
                2.70,
                0.85,
                0.08,
                0.003,
                2.0,
                1e-6,
                15.0,
                0.0031,
                30.0,
                1.0,
                0.999,
                0.0,
                1e-12,
                0.0,
                2.4e-9,
                1e-4,
                0.0,
            ]
        ),
    }

    def TheProjectileFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            theMaterialProjectile,
        )

    theModel = generateRectangularQuadParticleGrid(
        theModel,
        theJournal,
        TheProjectileFactory,
        x0=x0Projectile,
        y0=y0Projectile,
        h=heightProjectile,
        l=lengthProjectile,
        nX=nXProjectile,
        nY=nYProjectile,
        name="projectile",
    )

    # let's create the particle kernel domain
    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    # for Semi-Lagrangian particle methods, we assoicate a particle with a kernel function.
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True, neighbourListSkinFraction=0.0, rebuildShapeFunctionsEveryIncrement=True
    )

    # let's print some details
    print(theParticleManager)

    # We now create a bundled model.
    # We need this model to create the dof manager
    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain

    # Fix the bottom edge and pull the top edge by a prescribed displacement. Particles are selected
    # by coordinate rather than by generator set name, so the selection is explicit and checkable.
    allParticles = list(theModel.particles.values())

    def yOf(p):
        return p.getVertexCoordinates().mean(axis=0)[1]

    yMin = min(yOf(p) for p in allParticles)
    yMax = max(yOf(p) for p in allParticles)
    tol = 0.51 * particleSize
    bottomParticles = [p for p in allParticles if yOf(p) < yMin + tol]
    topParticles = [p for p in allParticles if yOf(p) > yMax - tol]
    print(f"constrained particles: {len(bottomParticles)} bottom, {len(topParticles)} top")

    def smootherstep(x):
        x = min(max(x, 0.0), 1.0)
        return x * x * x * (x * (6.0 * x - 15.0) + 10.0)

    prescribedEndDisplacement = 0.5 * heightProjectile * 0.0001  # 0.4 % nominal strain

    # Kinematic: no stiffness, so no cost to the stable increment. A monotonic ramp is a constant
    # velocity, so this is displacement control by another name.
    pullVelocity = prescribedEndDisplacement / tMax

    fixBottom = ParticlePrescribedVelocityExplicit(
        "fix_bottom", theModel, bottomParticles, "displacement", {0: 0.0, 1: 0.0}
    )
    pullTop = ParticlePrescribedVelocityExplicit("pull_top", theModel, topParticles, "displacement", {1: pullVelocity})
    print(f"prescribed velocity {pullVelocity:.4f} mm/s over {tMax:.3e} s -> {prescribedEndDisplacement:.4f} mm")

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["projectile_all"],
        "displacement",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "velocity",
        theModel.particleSets["all"],
        "velocity",
    )

    fieldOutputController.addPerParticleFieldOutput(
        "acceleration",
        theModel.particleSets["all"],
        "acceleration",
    )

    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["all"],
        "vertex displacements",
        reshape_to_dimensions=2,
    )
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    # ensightOutput.createPerElementOutput(fieldOutputController.fieldOutputs["displacement_projectile"])
    ensightOutput.createPerElementOutput(fieldOutputController.fieldOutputs["acceleration"])
    ensightOutput.createPerElementOutput(fieldOutputController.fieldOutputs["velocity"])
    ensightOutput.createPerNodeOutput(
        fieldOutputController.fieldOutputs["vertex displacements"], name="vertex displacements"
    )
    ensightOutput.createPerElementOutput(fieldOutputController.fieldOutputs["deformation gradient"])
    ensightOutput.initializeJob()

    incSize = 2e-3
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0, tMax, incSize, incSize, incSize / 1e8, 600, theJournal, increaseFactor=1.5
    )

    # nonlinearSolver = NQSParallelForMarmot(theJournal)
    nonlinearSolver = ExplicitMultiphysicsSolver(theJournal)

    # Quasi-static pull: no initial velocity.

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            prescribedVelocities=[fixBottom, pullTop],
            userIterationOptions={"field orders": {"displacement": 2, "nonlocal damage": 1}},
        )

    except ReachedMaxIncrements as e:
        theJournal.message(f"Reached maximum number of increments: {str(e)}", "error")

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        raise

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

    print(f"REACTION pull_top y: {pullTop.reactionHistory[-1][0]:.6e}")
    print(f"TOP DISP y: {_npmean(topParticles):.6e}  target {prescribedEndDisplacement:.6e}")

    import numpy as _np

    for _name in ("omega", "alphaP", "nonlocal damage"):
        try:
            _v = _np.array([_p.getResultArray(_name) for _p in theModel.particleSets["projectile_all"]])
            print(f"CHECK {_name:>16}: min={_v.min():.6e}  max={_v.max():.6e}  mean={_v.mean():.6e}")
        except Exception as _e:
            print(f"CHECK {_name:>16}: unavailable ({type(_e).__name__}: {_e})")

    return theModel, fieldOutputController


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):

    # disable plots and suppress warnings
    import matplotlib

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")

    theModel, fieldOutputController = run_sim()

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    gold = np.loadtxt("gold.csv")

    assert_gold(res, gold, atol=1e-12)


if __name__ == "__main__":
    theModel, fieldOutputController = run_sim()
    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())
