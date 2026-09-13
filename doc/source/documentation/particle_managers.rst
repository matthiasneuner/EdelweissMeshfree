Particle managers
=================

.. automodule:: edelweissmeshfree.particlemanagers.base.baseparticlemanager
   :members:
   :private-members:

Reusing neighbour lists: the skin
---------------------------------

Searching for the kernel functions that cover each particle is the single most expensive part of a
dynamic meshfree simulation, and in a dynamic simulation it is also largely redundant. Kernel functions
are bonded to particle centres, so a neighbour set changes through *deformation*, not through the body
as a whole moving: on an explicit Taylor bar impact, a mean of 0.41% of particles change their set from
one increment to the next, and most increments change none at all.

:class:`~edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager.KDBinOrganizedParticleManager`
can therefore be given a **neighbour list skin**. The search then reports every kernel function that
covers a particle *or comes within the skin of covering it*, and that answer is reused until accumulated
motion could have carried something across the remaining margin. Increments in between still rebuild the
shape functions, which have to be rebuilt in any case because the particles have moved.

This is exact rather than approximate. The kernel functions the skin admits early evaluate to exactly
zero at the particle, and the reproducing-kernel reconstruction discards anything that does, so the
shape functions are the ones a full search would have produced. A kernel function that *leaves* the
support is handled by the same mechanism. ``examples/144_neighbour_list_skin_test`` asserts this, by
running the same impact with the skin off and on and requiring the two displacement fields to be equal.

**This has only been shown to hold for single-field particles.** examples/144_neighbour_list_skin_test
(Displacement/RS-SNNIxNSNI/3D/Hexa) and a check against
examples/141_marmot_sqcni_nsni_finite_strain_plasticity_explicit_test
(Displacement/SQCNIxNSNI/PlaneStrain/Quad) both confirm bit-identical results with the skin on or
off. But the same check against a **gradient-enhanced** (two-field: displacement plus nonlocal damage)
particle -- GradientEnhancedFiniteStrainSQCNIxNSNI, as used by
examples/208_gradient_enhanced_explicit_dirichlet_test -- does *not* reproduce: turning on any
nonzero skin (0.005 or 0.05 tried, both giving the identical answer to each other) changes the final
reaction by 3.4% and the minimum damage variable omega by 14%, relative to searching every increment.
The size of the discrepancy does not shrink with the skin fraction, which rules out a simple accumulated
geometric error and points to the motion criterion missing something that matters for the second field --
most likely evaluation points relevant to the nonlocal damage field are not part of what
getEvaluationCoordinates() reports. This has not been root-caused. **Do not enable a nonzero skin for
a gradient-enhanced particle without first validating it for the specific case at hand**; the two worked
examples above are not evidence it will hold.

Choosing the skin is a trade-off, not a monotone improvement. A larger skin searches less often but
admits more kernel functions, and each of those costs something in the reconstruction and in the wider
degree-of-freedom stencil that the physics then has to touch. Measured on an explicit Taylor bar impact
at 108,864 degrees of freedom, over 20 increments:

=========================  ==================  ==================
``neighbourListSkin``      searches out of 21  time for the step
       ``Fraction``
=========================  ==================  ==================
0.0 (search every step)    22                  75.4 s
0.02                       7                   59.7 s
**0.05**                   **3**               **52.8 s**
0.10                       2                   63.7 s
=========================  ==================  ==================

So 0.05 is best here and 0.10 is worse than 0.05 despite searching less often. Since the optimum depends
on the impact velocity, the increment size, the support radius and how fast the body deforms, the
default is ``0.0`` -- every increment searches, exactly as before -- and a value has to be chosen
deliberately for the problem at hand.

A skin requires every kernel function to have box support, because the reuse argument relies on the
search testing an inflated bounding box; the manager refuses a non-zero skin otherwise rather than
silently over-reporting neighbours.

Going further: also skipping the rebuild
-----------------------------------------

The skin above only ever skips the *search*. Even on an increment where no search is due, the shape
functions are still rebuilt from the unchanged neighbour list every single time, because a
reproducing-kernel shape function is a function of the current position and the particles have moved.
Measured on a quasi-static explicit pull test (examples/208_gradient_enhanced_explicit_dirichlet_test,
single-threaded), the two halves of the connectivity update are comparably expensive -- roughly 48% search,
49% reconstruction, with the remainder in rebuilding the bins and moving bonded kernel functions -- so
skipping only the search leaves close to half of the phase's cost on the table.

``rebuildShapeFunctionsEveryIncrement=False`` skips the reconstruction too, whenever a search is not due.
This is **not exact** like the skin itself: the reused shape functions are stale by whatever motion has
accumulated since the last real rebuild, bounded by the same neighbour list skin but not zero. It trades a
small, bounded geometric error for skipping the more expensive half of the connectivity update on every
increment that the skin already made searchless. Off by default; opt in deliberately, and validate the
result against an unthrottled run for the problem at hand.

It inherits the skin's own limits: it only has an effect once neighbourListSkinFraction is nonzero,
and it cannot be safer than the skin it is built on -- see the gradient-enhanced caveat above, which
applies here too and is why it was not enabled for that example despite the further measured saving.

``KDBinOrganizedParticleManager`` class
---------------------------------------

.. automodule:: edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager
   :members:
   :private-members:

Legacy implementation
---------------------

.. automodule:: edelweissmeshfree.particlemanagers.oldkdbinorganizedparticlemanager
   :members:
   :private-members:

.. warning::

   rebuildShapeFunctionsEveryIncrement=False is **not** an approximation bounded by the
   neighbour-list skin, and in the most common meshfree configuration it is not bounded by anything
   the user sets.

   Reusing *neighbour lists* between searches is exact. Reusing *shape functions* is not: a
   reproducing-kernel shape function is a continuous function of position, so freezing it while the
   body deforms is a genuine geometric error. With bondParticlesToKernelFunctions=True the
   kernels travel with their particles and a search is therefore **never** due, which makes this flag
   mean *never rebuild after the initial build*.

   Measured on the gradient-enhanced bar of examples/208_gradient_enhanced_explicit_dirichlet_test:
   skins of 0.005 and 0.05 give **bit-identical** results, confirming the skin is inert here; the
   reaction moves 0.26 % while the response is elastic and 3.4 % once damage localises, with
   min(omega) moving 14 %; the connectivity phase drops 29x (12.85 s to 0.44 s).

   Use it where that trade is acceptable and verify it against a run with the default on the problem
   at hand. Do not use it for a softening or damage analysis whose results are to be quoted.
