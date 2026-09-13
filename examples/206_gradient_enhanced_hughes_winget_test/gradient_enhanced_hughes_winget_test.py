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

"""Drive the gradient-enhanced finite-strain meshfree POINT particle for the first time.

``GradientEnhancedFiniteStrain/PlaneStrain/Point`` (Marmot's ``GradientEnhancedFiniteStrainParticle``)
consumes a ``MarmotMaterialGradientEnhancedFiniteStrain``. ``GCDP/HUGHES-WINGET`` is the co-rotational
Hughes-Winget wrapper bridging Marmot's small-strain gradient-enhanced GCDP model into that interface,
exactly as ``VONMISES/HUGHES-WINGET`` does for the plain ``Displacement/PlaneStrain/Point`` particle in
``145_hughes_winget_wrapper_test``.

A small plane-strain tension bar is pulled in x until damage localises in a column of particles that
was seeded slightly weaker (``ftu`` lowered by 0.7 %), the standard imperfection trick for this kind of
localisation benchmark.
"""

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
)
from edelweissmeshfree.generators.rectangularparticlegridgenerator import (
    generateRectangularParticleGrid,
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


def run_sim():
    dimension = 2

    np.set_printoptions(linewidth=200)
    np.set_printoptions(precision=2)
    np.set_printoptions(threshold=np.inf)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    # Geometry: a plane-strain tension bar, pulled along x. Particle spacing 2 mm, nonlocal
    # radius R = lDamage = 5 mm = 2.5 spacings, bar length 60 mm = 12 R, height 20 mm = 4 R --
    # a few R long/tall in every direction, resolved by ~2-3 particles per R.
    x0 = 0.0
    y0 = 0.0
    spacing = 2.0
    length = 60.0
    height = 20.0
    nX = int(length / spacing) + 1
    nY = int(height / spacing) + 1
    supportRadius = 2.0 * spacing

    lDamage = 2.5 * spacing  # nonlocal radius R; c = R^2

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2)

    theModel = generateRectangularKernelFunctionGrid(
        theModel, theJournal, theMeshfreeKernelFunctionFactory, x0=x0, y0=y0, h=height, l=length, nX=nX, nY=nY
    )

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # GCDP properties, in order:
    # E, nu, fcy, fcu, fbu, ftu, Df, Ah, Bh, Ch, Dh, As, epsF, lDamage, m, maxDamage,
    # derivativeMethod, dTThreshold, viscosity, density, nonlocalViscosity, microInertia
    gcdpProperties = [
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
        lDamage,
        1.0,
        0.999,
        0.0,
        1e-12,
        0.0,
        2.4e-9,
        0.0,
        0.0,
    ]

    theMaterial = {
        "material": "GCDP/HUGHES-WINGET",
        "properties": np.array(gcdpProperties),
    }

    # a column of particles seeded 0.7 % weaker in ftu (index 5), to localise damage there
    weakGcdpProperties = list(gcdpProperties)
    weakGcdpProperties[5] *= 0.993
    theWeakMaterial = {
        "material": "GCDP/HUGHES-WINGET",
        "properties": np.array(weakGcdpProperties),
    }

    weakColumn = nX // 2

    def TheParticleFactory(number, coordinates, volume):
        # particles are created column-by-column (x-major, see generateRectangularParticleGrid),
        # so the column index follows from the running particle number
        columnIndex = (number - 1) // nY
        material = theWeakMaterial if columnIndex == weakColumn else theMaterial
        return MarmotParticleWrapper(
            "GradientEnhancedFiniteStrain/PlaneStrain/Point",
            number,
            coordinates,
            volume,
            theApproximation,
            material,
        )

    theModel = generateRectangularParticleGrid(
        theModel, theJournal, TheParticleFactory, x0=x0, y0=y0, h=height, l=length, nX=nX, nY=nY
    )

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )

    print(theParticleManager)

    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "nonlocal damage",
        theModel.particleSets["all"],
        "nonlocal damage",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "omega",
        theModel.particleSets["all"],
        "omega",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager("ensight", theModel, fieldOutputController, theJournal, None)
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["displacement"])
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["nonlocal damage"])
    ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["omega"])
    ensightOutput.initializeJob()

    targetDisplacement = 0.15  # mm, applied to the right face in x

    dirichletLeft = ParticlePenaltyWeakDirichlet(
        "left", theModel, theModel.particleSets["rectangular_grid_left"], "displacement", {0: 0.0}, 1e6
    )
    dirichletLeftBottomPin = ParticlePenaltyWeakDirichlet(
        "leftBottomPin",
        theModel,
        theModel.particleSets["rectangular_grid_leftBottom"],
        "displacement",
        {1: 0.0},
        1e6,
    )
    dirichletRight = ParticlePenaltyWeakDirichlet(
        "right",
        theModel,
        theModel.particleSets["rectangular_grid_right"],
        "displacement",
        {0: targetDisplacement},
        1e6,
    )

    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, 1e-2, 1e-2, 1e-5, 2000, theJournal)

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = dict()

    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 6
    iterationOptions["allowed residual growths"] = 3

    linearSolver = getLinSolverByName("pardiso", {})

    # instrument the right-face penalty constraint to record the reaction force (x direction)
    # at every converged increment, purely for reporting -- does not affect the solve.
    reactionForceHistory = []
    _originalApplyConstraint = dirichletRight.applyConstraint

    def _loggingApplyConstraint(dU, PExt, V, timeStep):
        result = _originalApplyConstraint(dU, PExt, V, timeStep)
        reactionForceHistory.append(dirichletRight.penaltyForce[0])
        return result

    dirichletRight.applyConstraint = _loggingApplyConstraint

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=[dirichletLeft, dirichletLeftBottomPin, dirichletRight],
            userIterationOptions=iterationOptions,
        )

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        raise

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

        if reactionForceHistory:
            theJournal.message(
                "peak |reaction force x| at right face: {:.6e}".format(max(abs(f) for f in reactionForceHistory)),
                "run_sim",
            )
        omegaResult = fieldOutputController.fieldOutputs["omega"].getLastResult()
        theJournal.message("max omega reached: {:.6f}".format(np.max(omegaResult)), "run_sim")

    return theModel, fieldOutputController


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):

    import matplotlib

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")

    theModel, fieldOutputController = run_sim()

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult().flatten()
    gold = np.loadtxt("gold.csv")

    assert_gold(res, gold)


if __name__ == "__main__":
    mpmModel, fieldOutputController = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        res = fieldOutputController.fieldOutputs["displacement"].getLastResult().flatten()
        np.savetxt("gold.csv", res)
