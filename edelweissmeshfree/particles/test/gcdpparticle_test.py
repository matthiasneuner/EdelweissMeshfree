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
"""A gradient-enhanced finite-strain particle of GCDP (``GCDP/HUGHES-WINGET``, one nonlocal variable) rotates its
stress with a rigid rotation, in the explicit scheme.

The Hughes-Winget wrapper is objective only for a material that is incremental in the stress: it rotates the stress of
the last increment and hands it to the material with the strain increment of the mid-step configuration. GCDP forms its
effective trial stress from that stress, and its other internal variables are scalars, so a particle that is stretched
until it is plastic and damaged and then rotated rigidly by 60 degrees in one increment must come out with the rotated
stress and the same scalar state. A material that rebuilt its stress from a stored strain would not (the AT2 trap of
the wrapper's documentation).

The particle is moved as an explicit analysis moves it: an increment of its nodes (``updatePhysicsExplicit``), accepted,
then its kernel functions follow their nodes (updated Lagrangian) and are assigned anew. GCDP is a private Marmot module;
skipped where it, or Marmot, is not built.
"""

import itertools

import numpy as np
import pytest
from edelweissfe.points.node import Node

#: GCDP: E, nu, fcy, fcu, fbu, ftu, Df, Ah, Bh, Ch, Dh, As, softening modulus, lDamage, m, omegaMax, derivative method,
#: dT threshold of the elastic tangent, Duvaut-Lions viscosity, density, nonlocal viscosity eta, micro-inertia eta^2/4
#: (the concrete of the FE-particle impact)
ETA = 2e-6
PROPERTIES = np.array(
    [30600.0, 0.2, 9.3, 32.3, 36.0, 2.7, 0.85, 0.08, 0.003, 2.0, 1e-6, 1.0, 0.0037, 4.0, 1.0, 0.995, 0.0, 1e-12]
    + [0.0, 2.4e-9, ETA, ETA**2 / 4]
)
H = 10.0  # the side of the cubic particle
DT = 1e-6


def _particleAndKernelFunctions():
    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )
    from edelweissmeshfree.particles.marmot import marmotparticlewrapper

    if marmotparticlewrapper.__file__.endswith(".py"):
        pytest.skip("Marmot is not built")

    vertices = H * np.array([[i, j, k] for k in (0, 1) for j, i in ((0, 0), (0, 1), (1, 1), (1, 0))], dtype=float)
    # raises NotImplementedError (a skip) where GCDP is not built
    particle = marmotparticlewrapper.MarmotParticleWrapper(
        "GradientEnhancedFiniteStrainSQCNIxNSNI/3D/Hexa",
        1,
        vertices,
        0.0,
        MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1),
        {"material": "GCDP/HUGHES-WINGET", "properties": PROPERTIES},
    )
    kernelFunctions = [
        MarmotMeshfreeKernelFunctionWrapper(
            Node(label, H * np.array(center)), "BSplineBoxed", supportRadius=1.5 * H, continuityOrder=3
        )
        for label, center in enumerate(itertools.product((-0.5, 0.5, 1.5), repeat=3))
    ]
    particle.assignKernelFunctions(kernelFunctions)
    particle.initializeYourself()
    return particle, kernelFunctions


def _moveTheNodes(particle, kernelFunctions, x: np.ndarray, dx: np.ndarray, dNonlocal: float, time: float):
    """One explicit increment: the nodes at x move by dx (one row per node) and the nonlocal field changes by dNonlocal
    at every node; the particle takes and accepts it, and the kernel functions follow their nodes."""

    dU = np.zeros((len(kernelFunctions), 4))
    dU[:, :3] = dx
    dU[:, 3] = dNonlocal
    particle.updatePhysicsExplicit(np.ascontiguousarray(dU.ravel()), time, DT)
    particle.acceptStateAndPosition()
    x += dx
    for kernelFunction, xA in zip(kernelFunctions, x):
        kernelFunction.moveTo(np.ascontiguousarray(xA))
    particle.assignKernelFunctions(kernelFunctions)


def _rotation(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def test_a_rigid_rotation_rotates_the_stress_of_a_damaged_gcdp_particle():
    particle, kernelFunctions = _particleAndKernelFunctions()
    assert particle.baseFields == ["displacement", "nonlocal damage"]
    x = np.array([kf.node.coordinates for kf in kernelFunctions], dtype=float)

    # an oblique uniaxial stretch to 0.5 %, with the nonlocal field following the local damage driving variable
    e = np.array([1.0, 0.5, -0.3]) / np.linalg.norm([1.0, 0.5, -0.3])
    F = np.eye(3)
    nonlocal_ = 0.0
    for i in range(1, 41):
        nextF = np.eye(3) + 1.25e-4 * i * np.outer(e, e)
        dNonlocal = particle.getResultArray("local damage")[0] - nonlocal_ if i > 1 else 0.0
        _moveTheNodes(particle, kernelFunctions, x, x @ (nextF @ np.linalg.inv(F) - np.eye(3)).T, dNonlocal, i * DT)
        F, nonlocal_ = nextF, nonlocal_ + dNonlocal

    scalarStates = ["alphaP", "alphaD", "omega", "nonlocal damage", "local damage"]
    before = {name: particle.getResultArray(name, getPersistentView=False).copy() for name in scalarStates}
    stressBefore = particle.getResultArray("stress", getPersistentView=False).reshape(3, 3)
    FBefore = particle.getResultArray("deformation gradient", getPersistentView=False).reshape(3, 3)
    assert before["alphaP"][0] > 0.0 and before["omega"][0] > 0.05, f"not plastic and damaged: {before}"
    np.testing.assert_allclose(FBefore, F, atol=1e-12)

    R = _rotation([0.3, -0.7, 0.65], np.pi / 3)
    _moveTheNodes(particle, kernelFunctions, x, x @ (R - np.eye(3)).T, 0.0, 41 * DT)

    np.testing.assert_allclose(particle.getResultArray("deformation gradient").reshape(3, 3), R @ FBefore, atol=1e-12)
    np.testing.assert_allclose(
        particle.getResultArray("stress").reshape(3, 3),
        R @ stressBefore @ R.T,
        rtol=0.0,
        atol=1e-9 * np.abs(stressBefore).max(),
    )
    for name in scalarStates:
        np.testing.assert_allclose(particle.getResultArray(name), before[name], rtol=1e-11, atol=1e-15, err_msg=name)
