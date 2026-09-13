import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePrescribedVelocityExplicit:
    """Kinematically prescribe a velocity on the kernel nodes supporting a set of particles.

    The counterpart of
    :class:`~edelweissmeshfree.constraints.explicit.particlepenaltyweakdirichletexplicit.ParticlePenaltyWeakDirichletExplicit`,
    and the reason it exists: **a penalty cannot drive an explicit meshfree analysis by a prescribed
    motion.** Measured on the tension bar of example 208, with the penalty the only stable stiffness
    (1e4) followed the prescribed displacement to 5.3 %, while the stiffness that would follow it to a
    few percent (>= 1e5) diverges. The explicit critical increment is computed from the material wave
    speed and takes no account of constraint stiffness, whereas a penalty's own limit falls off as
    :math:`1/\\sqrt{k}`, so the two requirements are irreconcilable.

    Imposing the motion kinematically introduces no stiffness at all, and therefore costs no increment.
    The explicit update is ``v_(n+1/2) += a_n dT`` followed by ``dU = v_(n+1/2) dT``; this overwrites both
    for the constrained degrees of freedom, so the prescribed motion is followed exactly, by construction,
    every step.

    For a monotonic ramp, prescribing a velocity **is** displacement control: reaching :math:`U` over a
    step of duration :math:`T` is :math:`\\bar v = U/T`.

    .. note::
       Reproducing-kernel shape functions are **not interpolatory**, so setting a few kernel nodes does
       not set the displacement of a particle sitting among them. This takes a set of *particles* and
       constrains **every kernel node supporting them**: the reproducing kernel reproduces constants
       exactly, so a uniform prescribed velocity over a contiguous node set is imposed exactly rather
       than approximately. Prescribing different velocities to overlapping node sets forfeits that.

    .. note::
       The reaction is not a constraint force here -- there is no spring -- but the internal force at the
       constrained degrees of freedom. Use :meth:`reaction` with the solver's internal force vector, or
       :meth:`recordReaction` to accumulate a history of it (see :attr:`reactionHistory`) once per
       finalized increment, the way :class:`ExplicitMultiphysicsSolver` samples it.

    Parameters
    ----------
    name
        The name of this boundary condition.
    model
        The full MPMModel instance.
    constrainedParticles
        The particles whose supporting kernel nodes are driven.
    field
        The field this boundary condition acts on.
    prescribedVelocity
        Field component -> prescribed velocity. Components absent from this dictionary are left free.
    f_t
        Optional shape of the prescribed history, a callable of the step progress, defaulting to a
        constant. Note that this scales the **velocity**, so a constant gives a linear displacement ramp.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedParticles: list[BaseParticle],
        field: str,
        prescribedVelocity: dict,
        **kwargs,
    ):
        self._name = name
        self._model = model
        self._constrainedParticles = constrainedParticles
        self._field = field
        self._prescribedVelocity = prescribedVelocity
        self._fieldSize = getFieldSize(field, model.domainSize)

        self._f_t = kwargs["f_t"] if "f_t" in kwargs else lambda x: 1.0

        self._cachedManager = None
        self._cachedSize = None
        self._idcsOfComponent = {}

        self._reactionHistory = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def reactionHistory(self) -> np.ndarray:
        """The reaction recorded by :meth:`recordReaction`, one row per call, one column per prescribed
        component, in the order of the prescribed dictionary. Empty until a driver records at least once.
        """
        return np.array(self._reactionHistory)

    def invalidate(self):
        """Forget the cached degree-of-freedom indices, forcing a rebuild on the next application."""
        self._cachedManager = None

    def _resolveIndices(self, theDofManager, nDof: int):
        """Map the supporting kernel nodes onto degree-of-freedom indices, once per dof structure."""
        if theDofManager is self._cachedManager and nDof == self._cachedSize:
            return

        nodes = sorted(
            set(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions),
            key=lambda n: n.label,
        )
        blocks = np.array([theDofManager.idcsOfFieldVariablesInDofVector[n.fields[self._field]] for n in nodes])

        self._idcsOfComponent = {c: blocks[:, c].copy() for c in self._prescribedVelocity}
        self._cachedManager = theDofManager
        self._cachedSize = nDof

    def applyKinematics(self, theDofManager, v: np.ndarray, dU: np.ndarray, dT: float, timeStep: TimeStep):
        """Overwrite the half-step velocity and the solution increment of the constrained dofs.

        Parameters
        ----------
        theDofManager
            The assembled DOF manager, for locating the constrained degrees of freedom.
        v
            The global half-step velocity vector, modified in place.
        dU
            The global solution increment vector, modified in place.
        dT
            The current time increment.
        timeStep
            The current time step, whose step progress drives the optional history shape.
        """
        self._resolveIndices(theDofManager, len(v))

        scale = self._f_t(timeStep.stepProgress)

        for component, prescribed in self._prescribedVelocity.items():
            idcs = self._idcsOfComponent[component]
            v[idcs] = prescribed * scale
            dU[idcs] = prescribed * scale * dT

    def reaction(self, theDofManager, P_Int: np.ndarray) -> np.ndarray:
        """The reaction carried by the constrained degrees of freedom.

        Parameters
        ----------
        theDofManager
            The assembled DOF manager.
        P_Int
            The global internal force vector.

        Returns
        -------
        np.ndarray
            One summed reaction per prescribed component, in the order of the prescribed dictionary.
        """
        self._resolveIndices(theDofManager, len(P_Int))

        return np.array([np.sum(P_Int[self._idcsOfComponent[c]]) for c in self._prescribedVelocity])

    def recordReaction(self, theDofManager, P_Int: np.ndarray):
        """Compute :meth:`reaction` and append it to :attr:`reactionHistory`.

        Meant to be called once per finalized increment -- the point at which the internal force vector
        reflects the state that was just accepted, and the same point at which field output is sampled --
        so that the recorded history lines up index-for-index with any per-increment output a driver
        script also collects (e.g. a particle's displacement), giving a load-displacement curve.

        Parameters
        ----------
        theDofManager
            The assembled DOF manager.
        P_Int
            The global internal force vector.
        """
        self._reactionHistory.append(self.reaction(theDofManager, P_Int))
