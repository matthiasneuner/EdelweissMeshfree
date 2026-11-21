Coupled RKPM + FE
=================

An :class:`~edelweissmeshfree.models.mpmmodel.MPMModel` may hold finite elements next to its particles: e.g. a
structure of finite elements with an RKPM domain where the material is expected to deform strongly. Both are solved
together, as one discrete system, by EdelweissFE's implicit static solvers (see :doc:`choosing_a_solver`). The time
integration is quasi-static.

Two kinds of nodes in one discrete system
-----------------------------------------

The model numbers the nodes of the finite elements and the nodes of the kernel functions in one discrete system,
the elements first, then the particles (:meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.buildDiscreteSystem`). The
two kinds of nodes hold different things:

- the nodes of the finite elements carry their **total** displacement from increment to increment, as in any finite
  element model;
- the nodes of the kernel functions carry the **coefficients of one increment**: RKPM is updated-Lagrangian, the
  kernel functions move with their particles, and the state lives in the particles. Every increment starts them at
  zero.

:class:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem` passes the solution between the
model and the solver's vectors accordingly. At acceptance, it writes the solution ``U``, the increment ``dU`` and the
reaction ``P`` straight to the model's node fields, by node: ``U`` is the total displacement at the nodes of the finite
elements and zero at the nodes of the kernel functions, whose meaningful value is ``dU``; every node outside the active
domain holds zero (see EdelweissFE's "Node field results" for the meaning of these entries).

The tie between the two domains
-------------------------------

The two domains do not share nodes. Sharing a node would not tie them: the reproducing kernel shape functions are not
interpolating, so the value at a kernel node is not the displacement there, while an element interprets the same
value as the displacement of its node. Instead, a constraint ties the two displacement fields along the interface,
:class:`~edelweissmeshfree.constraints.particlefiniteelementmortartie.ParticleFiniteElementMortarTie`:

- **in the weak (mortar) sense**: the difference of the displacements of the two sides, weighted by the shape functions
  of the element edges, integrates to zero. A displacement field that both sides represent exactly -- a linear one --
  is tied exactly;
- **with Lagrange multipliers**: one per interface node and component, the interface traction, interpolated like the
  displacement of the element edges. Unlike a penalty factor, they leave no gap and need no parameter;
- **incrementally, on the current configuration**: the tie couples the increments of both sides, on the configuration
  at the start of the increment, where RKPM defines its shape functions. The integrals are computed anew at the start
  of every increment, in ``updateConnectivity``, and the tie is linear during the increment.

The finite element side of the interface consists of the edges of quadratic elements, given by their nodes in order:
corner, midside, corner, ... The tie checks the order on the geometry: every midside node must lie near the middle of
its corners, and the interface must not turn back at a corner node. A node set in the order of its labels, with the
corner nodes numbered before the midside nodes, is refused. The RKPM side is given by the particles along the
interface. The interface nodes of the finite elements carry the multipliers; they must not carry a Dirichlet boundary
condition too.

.. code-block:: python

    from edelweissmeshfree.constraints.particlefiniteelementmortartie import ParticleFiniteElementMortarTie

    model.constraints["tie"] = ParticleFiniteElementMortarTie(
        "tie", model, model.particleSets["rkpm_left"], model.nodeSets["fe_right"]
    )

The constraint must be part of the model before ``model.prepareYourself``, which creates the multipliers.

The mortar tie is for implicit analyses in 2D: an explicit solver does not take multipliers. In 3D, and in explicit
analyses, a **penalty tie** joins the two domains: EdelweissFE's contact constraint with the law ``type=tied``, whose
contact points are the quadrature points of the faces of the particles at the interface
(:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`) and whose
master is the faceted surface of the elements there. It ties each point, once, to the closest point of the elements'
faces, keeps a gap or an overlap between the two meshes as the state at rest, and holds the point there by springs of
the penalty stiffness per unit area -- so the interface is as compliant as the springs, and the springs are as stiff as
the time increment of an explicit analysis allows. See :doc:`constraints`, *Contact of particles with finite elements*,
and examples 194 (the patch test) and 195 (a pulse through the ties).

.. code-block:: python

    from edelweissfe.constraints.surfacetodeformablesurfacepenalty import Constraint, SurfaceToDeformableSurfacePenaltySchema
    from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import ContactPointsOnParticleFaces

    model.constraints["tie"] = Constraint(
        "tie", model, ContactPointsOnParticleFaces(model.surfaces["rkpm_bottom"], nQuadraturePoints=4),
        model.elementSets["fe_top_facets"], journal,
        configuration=SurfaceToDeformableSurfacePenaltySchema(penalty=1e6, contactType="tied", searchDistance=0.5),
    )

Building a coupled model
------------------------

The finite elements come from EdelweissFE's generators, inside a topology change; the kernel functions get node
labels after those of the elements; the sections of the elements are assigned by ``model.prepareYourself``:

.. code-block:: python

    with model.topology.changes():
        PlaneRectQuadGenerator(
            "fe", model, journal,
            configuration=PlaneRectQuadSchema(x0=0.0, length=3.0, h=1.0, nX=6, nY=2, elType="CPE8UL"),
        )
    model.sections["fe"] = PlaneSection(
        "fe", model, {"name": "CompressibleNeoHooke", "properties": materialProperties},
        [model.elementSets["fe_all"]], configuration=PlaneSectionSchema(thickness=1.0),
    )

    generateRectangularKernelFunctionGrid(model, journal, kernelFunctionFactory, ...,
                                          firstKernelFunctionNumber=max(model.nodes) + 1)

Dirichlet boundary conditions on finite element nodes are EdelweissFE's ``dirichlet`` step actions, as in a finite
element model.

The example
-----------

``examples/170_coupled_rkpm_fe_cantilever_test`` is a cantilever whose middle part is an RKPM domain between two parts
of quadratic finite strain elements, clamped and deflected at finite element nodes, and solved by ``NISTParallel``.
Its independent check is the same cantilever made of finite elements only: the coupled model reproduces its tip
reaction and its deflection line, and both approach the same limit as the elements and particles are refined.
