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
"""The two operations every constraint on points of particles needs besides its own physics: where the nodes of its
particles sit in its block, and how a force on a point of a particle reaches those nodes.

A constraint that measures a point of a particle -- a vertex held by a support, a vertex in contact -- acts on the nodes
of the particle's kernel functions, which change as the particle moves. Its block holds each of these nodes once
(:func:`locateTheNodesOfParticles`). A force on the point reaches the nodes with the vector the constraint chose for
the point (:func:`addForcesOnPointsOfParticle`;
:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`): the
shape functions at the point, or the particle's interpolation linearized about its center, with which it moves the
point, the transpose of the point's motion, so that a penalty on the point has a symmetric stiffness and does no work
over a closed cycle. The explicit solver bounds that stiffness for its critical time step
(:func:`addBoundOfAbsoluteStiffnessOfPointsOfParticle`, the one bound of penalty springs of EdelweissFE,
:mod:`edelweissfe.constraints.base.penaltysprings`). A penalty constraint gives its springs on the points of a particle
as :class:`SpringsOnPointsOfParticle`, from one method for its force and its bound, so that the two cannot drift
apart. The constraints of :mod:`edelweissmeshfree.constraints.explicit` and
:class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet` use these; a new one
is then its physics -- the gap, the pairing, the force -- and a constructor.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from edelweissfe.constraints.base.penaltysprings import (
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
)


def locateTheNodesOfParticles(kernelFunctionsOfParticles: Iterable[list], localIndexOfNode: dict) -> list[np.ndarray]:
    """Locate the nodes of the kernel functions of particles in the block of a constraint: a node not located yet is
    added to ``localIndexOfNode`` at the next index, in the order in which the particles name their kernel functions;
    a node shared by several particles is located once.

    Parameters
    ----------
    kernelFunctionsOfParticles
        The kernel functions of each particle.
    localIndexOfNode
        Where each node located so far sits in the block of the constraint; extended here. The keys, in their order,
        are the nodes of the constraint.

    Returns
    -------
    list[numpy.ndarray]
        Per particle, where the nodes of its kernel functions sit in the block, in the order of its kernel functions.
    """

    # indices also for a particle without kernel functions: an empty array without a type would be one of floats
    return [
        np.array(
            [localIndexOfNode.setdefault(kernelFunction.node, len(localIndexOfNode)) for kernelFunction in kfs],
            dtype=np.intp,
        )
        for kfs in kernelFunctionsOfParticles
    ]


def interpolationVectorAtPointOfParticle(particle, coordinates: np.ndarray, linearizedAboutCenter: bool) -> np.ndarray:
    """The vector with which a constraint measures a field at a point of a particle and distributes its force there:
    the particle's interpolation linearized about its center, :math:`\\mathbf{G}`, if ``linearizedAboutCenter``, else the
    shape functions at the point, :math:`\\mathbf{N}`. What each means for a constraint is documented once, at
    :meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`,
    which the constraints call; the contact points on the faces of particles
    (:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`), which are
    no constraint of their own, call this function.

    Parameters
    ----------
    particle
        The particle.
    coordinates
        The point, in the current configuration of the particle.
    linearizedAboutCenter
        The option ``interpolationLinearizedAboutCenter`` of the constraint or of the contact points.

    Returns
    -------
    np.ndarray
        The vector, one value per kernel function of the particle.
    """

    if linearizedAboutCenter:
        return particle.getInterpolationVectorLinearizedAboutCenter(coordinates)
    return particle.getInterpolationVector(coordinates)


def addForcesOnPointsOfParticle(
    PAtNodes: np.ndarray,
    localNodes: np.ndarray,
    interpolations: np.ndarray,
    forces: np.ndarray,
    components: list[int] = None,
):
    """Add the residual contribution of forces on points of one particle, the negative of the forces on the nodes of its
    kernel functions: :math:`\\boldsymbol{P}_B \\mathrel{-}= \\sum_p I_{pB}\\, \\boldsymbol{f}_p`, with the vector
    :math:`I_{pB}` the constraint distributes the force on the point :math:`p` with
    (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`).

    Parameters
    ----------
    PAtNodes
        The residual contribution of the constraint, node by node: ``nNodes x fieldSize``; augmented.
    localNodes
        Where the nodes of the particle's kernel functions sit in the block (:func:`locateTheNodesOfParticles`). They
        are distinct, so the fancy indexing drops no contribution.
    interpolations
        The vectors of the points, one row per point: ``nPoints x nNodesOfTheParticle``.
    forces
        The forces on the points, one row per point: ``nPoints x fieldSize``, or ``nPoints x len(components)``.
    components
        The components the forces act in, e.g. those a support holds; all if None.
    """

    if components is None:
        PAtNodes[localNodes] -= interpolations.T @ forces
    else:
        PAtNodes[np.ix_(localNodes, components)] -= interpolations.T @ forces


@dataclass(frozen=True)
class SpringsOnPointsOfParticle:
    """Penalty springs on points of one particle, each along a unit direction: spring :math:`p` pulls its point along
    :math:`\\boldsymbol{n}_p` with the force :math:`f_p`, which reaches the nodes with the interpolation vector of the
    point, :math:`\\boldsymbol{f}_p = f_p \\boldsymbol{n}_p` in :func:`addForcesOnPointsOfParticle`."""

    #: The vectors of the points of the springs, one row per spring: ``nSprings x nNodesOfTheParticle``, those the
    #: constraint measures and distributes with
    #: (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`).
    interpolations: np.ndarray
    #: The unit direction of each spring: ``nSprings x fieldSize``.
    directions: np.ndarray
    #: The force of each spring along its direction.
    forces: np.ndarray
    #: The stiffness of each spring, :math:`k_p = \\mathrm{d}f_p/\\mathrm{d}g_p`.
    stiffness: np.ndarray


def addBoundOfAbsoluteStiffnessOfPointsOfParticle(
    bound: np.ndarray, weights: np.ndarray, localNodes: np.ndarray, springs: SpringsOnPointsOfParticle
):
    """Add the row sums of the absolute value of the stiffness of penalty springs on points of one particle, weighted
    by column (see :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`).
    A spring of stiffness :math:`k_p` on the point :math:`p` along the unit direction :math:`\\boldsymbol{n}_p` has the
    stiffness :math:`k_p\\, \\boldsymbol{w}_p \\otimes \\boldsymbol{w}_p` with :math:`\\boldsymbol{w}_p = I_p
    \\otimes \\boldsymbol{n}_p` and the vector :math:`I_p` of the point (:attr:`SpringsOnPointsOfParticle.interpolations`),
    so

    .. math::

        b_{Bi} \\mathrel{+}= \\sum_p |k_p|\\, |I_{pB}|\\, |n_{pi}| \\sum_{C,j} |I_{pC}|\\, |n_{pj}|\\, w_{Cj} ,

    the bound of every penalty spring
    (:func:`~edelweissfe.constraints.base.penaltysprings.addBoundOfAbsoluteStiffnessOfSprings`), over the nodes of the
    particle's kernel functions.

    Parameters
    ----------
    bound
        The row sums of the constraint, one per degree of freedom of its block (node by node); augmented.
    weights
        The weights of the columns, likewise.
    localNodes
        Where the nodes of the particle's kernel functions sit in the block (:func:`locateTheNodesOfParticles`).
    springs
        The springs.
    """

    fieldSize = springs.directions.shape[1]
    dofsOfNodes = localNodes[:, None] * fieldSize + np.arange(fieldSize)
    addBoundOfAbsoluteStiffnessOfSprings(
        bound,
        weights,
        springs.stiffness,
        [SpringGradientOnSharedNodes(springs.interpolations, springs.directions, dofsOfNodes)],
    )
