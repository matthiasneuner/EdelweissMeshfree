Constraints
===========

.. _residual-and-tangent-convention:

The residual and the tangent
----------------------------

A constraint of EdelweissMeshfree is an EdelweissFE constraint, and enters the discrete system by EdelweissFE's rule
(see the EdelweissFE solver documentation, *The residual and the tangent*): its ``computeKernels(U_np, dU, P, K,
timeStep)`` adds the residual contribution :math:`\mathbf{P}` -- the *negative* of the force the constraint exerts on
its nodes, what an element of the same mechanics would add as its internal force -- and its tangent
:math:`\mathbf{K} = \partial \mathbf{P} / \partial \mathbf{U}`. A penalty constraint that pulls the displacement
:math:`u = \mathbf{N} \cdot \Delta\mathbf{U}` of a particle towards :math:`\bar{u}` adds

.. math::

    \mathbf{P} \mathrel{+}= k\, \mathbf{N}\, (\mathbf{N} \cdot \Delta\mathbf{U} - \bar{u}), \qquad
    \mathbf{K} \mathrel{+}= k\, \mathbf{N} \otimes \mathbf{N} .

The nodal values of the particles and of the cells are increments on the current configuration, so these constraints
read the increment ``dU``. Their nodes are those the particles or material points currently interpolate from: a
constraint finds them anew at the start of every increment, in ``updateConnectivity``, which may also find the
constraint out of reach (``withinReach``), e.g. a contact constraint whose particle is far from the boundary. A
constraint takes part in the discrete system (``active``) while it is within reach, unless it is switched off from
outside: setting ``active`` to False, e.g. by a ``modelupdate`` step action, switches a constraint off whatever it
finds in reach, until it is switched on again. Switched off from outside, it rests: the model does not locate it anew,
and it keeps what it found and computed last, its reaction force included. These are the rules of EdelweissFE's
constraint base class, see the EdelweissFE constraint documentation.

A constraint with Lagrange multipliers adds them as scalar variables of the model. The condition :math:`g` it enforces
is one on increments, like the nodal values; the multiplier :math:`\lambda` is the force or traction that balances the
total stresses of the particles or elements, so it is a **total**. A multiplier that prescribes the increment
:math:`\Delta\bar{u}` of a particle's displacement :math:`u = \mathbf{N} \cdot \Delta\mathbf{U}` adds

.. math::

    \mathbf{P} \mathrel{+}= \lambda\, \mathbf{N}, \qquad
    P_{\lambda} \mathrel{+}= \mathbf{N} \cdot \Delta\mathbf{U} - \Delta\bar{u}, \qquad
    \mathbf{K}_{U\lambda} \mathrel{+}= \mathbf{N}, \qquad
    \mathbf{K}_{\lambda U} \mathrel{+}= \mathbf{N}^T,

with :math:`\lambda` read from ``U_np``: the solver continues it from the last converged increment, also across a
rebuilt discrete system, as the model keeps the converged multipliers in its scalar variables, and its checkpoint keeps
them for a restart. The residual is linear in :math:`\lambda`, so where Newton's method starts :math:`\lambda` changes
only its first residual, not the iterates after the first correction. The deprecated solvers
(:mod:`edelweissmeshfree.solvers.deprecated`) continue the multipliers only while the discrete system is kept, restart
them at zero after a rebuild, and do not check the residual of the multipliers for convergence.

Because :math:`\mathbf{K}` is the derivative of exactly what is added to :math:`\mathbf{P}`, each constraint is
checked by central differences of its own ``computeKernels`` in
``edelweissmeshfree/constraints/test/residual_tangent_test.py``, with EdelweissFE's check
(:mod:`edelweissfe.utils.residualtangentcheck`); add a case there for a new one.

A constraint written for explicit analyses alone, with forces and without a tangent, derives from
:class:`~edelweissmeshfree.constraints.base.mpmexplicitconstraintbase.MPMExplicitConstraintBase` and implements
``computeKernelsExplicit(U_np, dU, P, timeStep)``, with the same sign: an implicit solver refuses it. The explicit
solver, in turn, accepts only a constraint that acts in an explicit analysis (``actsInExplicitAnalysis``): one that
measures its condition at the particles' current positions, as the penalty contacts do, explicit or not, and as the
penalty weak Dirichlet condition does where it holds the displacement at zero (a support). A constraint on the
increment, such as a weak Dirichlet condition that prescribes a change, would act as if nothing had moved there.

**Points of a particle.** A particle knows every field only by its value and its gradient at its center, so its own
interpolation at a point of its own -- a vertex, a face center, its center -- is its shape functions linearized about
the center, :math:`\mathbf{G}`
(:meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVectorLinearizedAboutCenter`): for
the displacement, the motion of the point. The shape functions at the point, :math:`\mathbf{N}`
(:meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVector`), are exact at an arbitrary
location and differ from :math:`\mathbf{G}` away from the center. The constraints that act on points of particles (the
penalty weak Dirichlet condition, the cartesian boundaries, the rigid body contacts, implicit and explicit) and the
contact points on the faces of particles (:ref:`the contact, the tie and the rigid contact of particle faces
<particle-face-contact>`) take one option, ``interpolationLinearizedAboutCenter``
(:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`):

- ``False``, the default: they measure the field at the point and distribute their force there with :math:`\mathbf{N}`.
  In an explicit analysis, where the point is where the particle has moved it with :math:`\mathbf{G}`, the penalty
  stiffness :math:`k\,\mathbf{N}\mathbf{G}^\mathsf{T}` is unsymmetric: it does work over a closed cycle of motion and
  can feed energy into softened material next to a support.
- ``True``: they use :math:`\mathbf{G}` for both, for every field: the stiffness
  :math:`k\,\mathbf{G}\mathbf{G}^\mathsf{T}` is symmetric and the constraint conservative;
  ``edelweissmeshfree/constraints/test/vertexpenaltywork_test.py`` checks that a support on the vertices and a contact
  on a face of a real particle then do no work over a closed cycle.
  The FE-particle impact (:doc:`fe_projectile_on_rkpm_slab`) and the examples of the contact and the tie of particle
  faces (185, 187, 191-195) opt in.

The two operations a constraint on points of particles needs besides its own
physics -- locating the nodes of its particles in its block, and distributing a force on a point -- are in
:mod:`edelweissmeshfree.constraints.base.pointsofparticles`, which the explicit constraints and the support use: a new
constraint on points of particles is its physics and a constructor.

The base classes
----------------

.. automodule:: edelweissmeshfree.constraints.base.mpmconstraintbase
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.constraints.base.mpmexplicitconstraintbase
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.constraints.base.pointsofparticles
   :members:

``PenaltyEqualValue`` constraint class
--------------------------------------

.. automodule:: edelweissmeshfree.constraints.penaltyequalvalue
   :members:
   :private-members:

``PenaltyConstrainMP2Node`` constraint class
--------------------------------------------

.. automodule:: edelweissmeshfree.constraints.penaltyconstrainmp2node
   :members:
   :private-members:

``PenaltyWeakDirichlet`` constraint class
-----------------------------------------

.. automodule:: edelweissmeshfree.constraints.penaltyweakdirichlet
   :members:
   :private-members:


``ParticlePenaltyWeakDirichlet`` constraint class
-------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltyweakdirichtlet
   :members:
   :private-members:

``ParticlePenaltyEqualValue`` constraint class
----------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltyequalvalue
   :members:
   :private-members:

``ParticlePenaltyCartesianBoundary`` constraint class
------------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltycartesianboundary
   :members:
   :private-members:

``ParticlePenaltyRigidBodyContact`` constraint class
----------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltyrigidbodycontact
   :members:
   :private-members:

``ParticleLagrangianEqualValue`` constraint class
--------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlelagrangianequalvalue
   :members:
   :private-members:

``ParticleLagrangianWeakDirichlet`` constraint class
-----------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlelagrangianweakdirichlet
   :members:
   :private-members:

``ParticleLagrangianCartesianBoundary`` constraint class
---------------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlelagrangiancartesianboundary
   :members:
   :private-members:

``ParticleMortarWeakDirichlet`` constraint class
-------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlemortarweakdirichlet
   :members:
   :private-members:

``ParticleFiniteElementMortarTie`` constraint class
---------------------------------------------------

Ties an RKPM domain to a finite element domain, see :doc:`coupled_rkpm_fe`. Its tangent is checked by central
differences in ``edelweissmeshfree/constraints/test/mortar_tie_test.py``, together with the integrals it stands for.

.. automodule:: edelweissmeshfree.constraints.particlefiniteelementmortartie
   :members:
   :private-members:

Constraints for explicit analyses
---------------------------------

``ParticlePenaltyCartesianBoundaryConstraintExplicit`` constraint class
***********************************************************************

.. automodule:: edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit
   :members:
   :private-members:

``ParticlePenaltyContactImplicitSurfaceConstraintExplicit`` constraint class
****************************************************************************

.. automodule:: edelweissmeshfree.constraints.explicit.particlepenaltyrigidbodycontactexplicit
   :members:
   :private-members:

.. _particle-face-contact:

Contact of particles with finite elements
-----------------------------------------

Particles meet a body of finite elements through EdelweissFE's contact constraint,
:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`, the same one that couples two bodies of finite
elements: it is one contact for every kind of contact point (see EdelweissFE's contact theory, *One contact for every
kind of contact point*). The contact points of particles are quadrature points on faces of particles -- a surface of
particles, as the generators and the side sets of an Exodus mesh make them, or (particle, face) pairs -- each moving
with the particle's interpolation linearized about its center, :math:`\mathbf{G}`, with
``interpolationLinearizedAboutCenter`` (see *Points of a particle* above), and standing for its share of the face, so the penalty is a stiffness per unit area, as for finite elements. The master is the faceted boundary of the finite
elements, which EdelweissFE's surface element generator creates; it may have corners, convex and re-entrant, such as
the crests and roots of a thread (EdelweissFE's contact theory, *Corners of the master*). An overlap at the start is
either real, such as a groove cut smaller than the thread, and pushed apart, at once or over a time
(``initialOverlap=pushApart``, the default), or an error of the geometry, and adjusted: kept as the stress-free state
(``initialOverlap=adjust``; *An initial overlap*). The flat faces of particles in the curved hole of a screw are the
second kind: they are chords of the screw's arcs and lie inside the screw by their sag, although the concrete fits the
screw it was cut by. The contact acts in implicit and explicit analyses;
its tests are ``edelweissmeshfree/constraints/test/particlefacecontact_test.py`` (the force of a pressed face, the
tangent by central differences, pairing, entering, tunnelling, an overlap at the start ramped out or adjusted,
restart) and examples 185 (the same result as an
analytic wall), 187 (impact of particles and of finite elements on finite elements), 191
(:doc:`fe_projectile_on_rkpm_slab`), 192 (the contact patch test, implicit: the pressure a block of particles transmits
between two platens of finite elements, against a block of finite elements) and 193 (a toy pull-out: the head of a stud
of finite elements bearing on particles, implicit).

The same constraint with the law ``type=tied`` **ties** particles to finite elements instead (EdelweissFE's contact
theory, *A tie*): each point of a face of a particle is paired once with the closest point of the faces of the
elements, and held there in every direction by penalty springs, in tension and compression; a gap or an overlap of the
two surfaces at the pairing -- two meshes of one curved surface never coincide -- is part of the state at rest. It
joins a region of particles to a far field of finite elements, in 3D, in implicit and explicit analyses, and keeps its
state when the elements are refined. Its tests are in the same file (at rest with a gap and with an overlap, pulled
back in every direction, the tangent, a particle that moves on) and examples 194 (the patch test of the tie, implicit:
a column of particles between two blocks of elements, stretched and sheared, with a gap or an overlap at the start) and
195 (a pulse through a bar of elements with a tied segment of particles, explicit, against the bar of elements alone).
Two things to know: the traction of a tie at a single point of particles follows the particles' small deviations from
the exact displacement, magnified by the stiff springs -- its resultant over a few particles is the reliable measure;
and the springs' stiffness raises the frequencies of the points they hold, which ``NED``'s critical time step counts
(its springs along the axes, like the contact's normal spring; :doc:`choosing_a_solver`).

Gradient-enhanced particles tied to gradient-enhanced elements need their **nonlocal field tied** as well, or it ends
at the tie with no flux, and the damage with it: ``tiedFields="displacement; nonlocal damage"`` holds the field equal
on both sides by a penalty :math:`\beta l^2/h` per unit area (EdelweissFE's contact theory, *A nonlocal field across
the tie*; ``nonlocalRadius`` :math:`l`, ``nonlocalPenaltyFactor`` :math:`\beta`). The nodes of the kernel functions
carry the increment of the field; its value at a point of a particle at the update is the particle's own, from the
value and the gradient at its center that the particle keeps in its state (``nonlocal damage``, ``nonlocal damage
gradient``), at the offset of the point from the center
(:meth:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces.valuesOfAFieldAtTheOriginsOfLocatedPoints`).
The tests (``constraints/test/particlefacetiefield_test.py``): a linear field crosses the tie with no force, from the
increment at the nodes and, once accepted, from the particles' state, which equals what the increment added up to; a
uniform jump gives equal and opposite fluxes; the tangent; and the critical time step of the field with the tie's rows,
first and second order, below the exact eigenvalue. The tie of the field acts on the nonlocal field of the particles,
so under ``carry-critical-time-step-weights`` the particles keep their weights constant on that field, as on the
displacement.

Contact of particles with a rigid body
--------------------------------------

The same contact points meet a discrete rigid body -- a support, an indenter, a projectile that does not deform --
through EdelweissFE's :mod:`edelweissfe.constraints.surfacetodiscreterigidbodypenalty`, the contact of the faces and
nodes of finite elements with a rigid body (EdelweissFE's contact theory, *Integrated contact against a discrete rigid
body*): the constraint is built with :class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`
in place of the faces of elements. Each point is paired with the plane of its closest triangle of the rigid surface,
which moves and rotates with the body's reference point; the force of a point reaches the nodes of the kernel functions
of its particle with :math:`\mathbf{G}` (or :math:`\mathbf{N}`, by ``interpolationLinearizedAboutCenter``), and the
reference point with the force and its moment. Points farther than the
search distance from the rigid surface take no part until they approach. Its tests are
``edelweissmeshfree/constraints/test/particlefacerigidcontact_test.py`` (the force and the moment of a pressed face,
action = reaction, the tangent by central differences against a moved and rotated body, a particle that approaches).
The particle constraints for analytical surfaces above (``ParticlePenaltyRigidBodyContact``, with an implicit function)
remain the way to describe a boundary by a formula rather than by a mesh.

.. automodule:: edelweissmeshfree.constraints.base.contactpointsonparticlefaces
   :members:
