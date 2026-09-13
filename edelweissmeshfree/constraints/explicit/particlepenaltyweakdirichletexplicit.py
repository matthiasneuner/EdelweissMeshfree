import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyWeakDirichletExplicit(MPMConstraintBase):
    """Weak Dirichlet boundary conditions for EXPLICIT simulations, in a penalty formulation.

    The explicit counterpart of
    :class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet`,
    and the reason it is needed: until now nothing on the explicit meshfree path could prescribe a
    displacement. The explicit solver has no ``constraints`` argument on ``solveStep`` and passes its
    model-level constraints neither a solution increment nor a system matrix, so the implicit
    constraint cannot be reused -- every explicit meshfree analysis in this code base has been
    impact- or load-driven instead.

    Two differences from the implicit formulation follow from that:

    - It is written in **total** form. The implicit constraint prescribes the increment of the field
      over one solver increment and is handed ``dU`` to measure it against; an explicit step has no
      such increment to iterate on, so the target here is the accumulated displacement
      :math:`u_{\\mathrm{target}}(t) = u_0 + \\Delta u_{\\mathrm{step}}\\, f(\\tau)` and the measured
      value is the particle's own accumulated displacement.
    - It contributes a **restoring force to the external force vector**, with the sign the explicit
      solver's ``Rhs = P_ext - P_int`` requires, rather than a flux-side contribution plus a
      stiffness.

    The penalty stiffness enters the stable time increment as any other stiffness does. It has to be
    stiff enough to hold the boundary and soft enough not to dominate the critical increment, and the
    solver's reported critical increment is the thing to watch when choosing it.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    constrainedParticles
        The list of particles to be constrained.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        The prescribed change of the field components over the present load step.
    penaltyParameter
        The penalty parameter value.
    constrain
        Either ``"center"`` to constrain the particle centre, or a list of vertex indices for
        particles with multiple vertices.
    f_t
        Optional shape of the prescribed history over the step, a callable of the step progress.
        Defaults to a linear ramp. A smooth ramp is usually the better choice explicitly, since a
        linear one starts with a velocity jump.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedParticles: list[BaseParticle],
        field: str,
        prescribedStepDelta: dict,
        penaltyParameter: float,
        constrain: str | list[int] = "center",
        **kwargs,
    ):
        self._name = name
        self._model = model
        self._constrainedParticles = constrainedParticles
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter
        self._nodes = dict()

        if constrain == "center":
            self._constrainVertices = None
        else:
            self._constrainVertices = constrain

        # The measured reaction, so a load-displacement curve can be recovered from an explicit run.
        self.penaltyForce = np.zeros(self._fieldSize)

        self._f_t = kwargs["f_t"] if "f_t" in kwargs else lambda x: x

        # Baseline displacement of every constrained evaluation point, captured on the first
        # evaluation of the step. The prescribed delta is measured from it, so that a second step
        # continues from where the first ended rather than snapping back to the reference state.
        self._u0 = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes.keys()

    @property
    def fieldsOnNodes(self) -> list:
        return [
            [
                self._field,
            ]
        ] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list[ScalarVariable]:
        return []

    @property
    def active(self) -> bool:
        return True

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def resetBaseline(self):
        """Forget the captured baseline, so the next evaluation captures it afresh."""
        self._u0 = None

    def updateConnectivity(self, model):
        nodes = {
            n: i
            for i, n in enumerate(
                sorted(
                    set(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions), key=lambda n: n.label
                )
            )
        }

        hasChanged = nodes != self._nodes

        self._nodes = nodes

        return hasChanged

    def _constrainedCoordinatesOf(self, p: BaseParticle) -> list:
        """The points of one particle at which the constraint is enforced."""
        if self._constrainVertices:
            return p.getVertexCoordinates()[self._constrainVertices]
        return [p.getCenterCoordinates()]

    def _captureBaseline(self):
        """Record the accumulated displacement the prescribed delta is measured from."""
        self._u0 = [np.array(p.getResultArray("displacement"), copy=True) for p in self._constrainedParticles]

    def applyConstraint(self, PExt: np.ndarray, timeStep: TimeStep):
        """Add the penalty restoring force of every constrained particle to the external force vector.

        Parameters
        ----------
        PExt
            The constraint's local external force vector, of size :attr:`nDof`.
        timeStep
            The current time increment; its step progress drives the prescribed history.
        """
        if self._u0 is None:
            self._captureBaseline()

        ramp = self._f_t(timeStep.stepProgress)

        for i, prescribedComponent in self._prescribedStepDelta.items():
            P_i = PExt[i :: self._fieldSize]

            for p, u0 in zip(self._constrainedParticles, self._u0):
                uTarget = u0[i] + prescribedComponent * ramp
                uActual = p.getResultArray("displacement")[i]

                # The restoring force opposes the deviation, and goes to the external force vector
                # with the sign the explicit update Rhs = P_ext - P_int needs.
                forceScalar = -self._penaltyParameter * (uActual - uTarget)

                nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

                for constrainedCoordinate in self._constrainedCoordinatesOf(p):
                    N = np.asarray(p.getInterpolationVector(constrainedCoordinate)).flatten()
                    P_i[nodeIdcs] += N * forceScalar

            self.penaltyForce[i] = np.sum(P_i)
