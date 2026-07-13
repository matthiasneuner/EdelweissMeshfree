from collections.abc import Callable, Iterable

import numpy as np
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

from edelweissmeshfree.constraints.particlepenaltyrigidbodycontact import (
    ParticlePenaltyContactImplicitSurfaceConstraint,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class FrictionalParticlePenaltyContactImplicitSurfaceConstraint(ParticlePenaltyContactImplicitSurfaceConstraint):
    """
    Frictional penalty contact constraint for arbitrary implicit surfaces.

    Uses a Coulomb stick-slip friction model with a tangential penalty
    (elastic predictor / radial return). The stick force committed at the end
    of the previous increment is stored per contact point; Newton iterations
    compute a trial force from that committed state and never mutate it, so
    the result is independent of the iteration count. The trial state is
    committed upon increment acceptance via :meth:`acceptLastState`.
    """

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        implicit_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        model: MPMModel,
        location: str = "center",
        faceIDs: list[int] = None,
        vertexIDs: list[int] = None,
        penaltyParameter: float = 1e5,
        frictionCoefficient: float = 0.0,
        tangentialPenaltyParameter: float = None,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
    ):
        super().__init__(
            name=name,
            particle=particle,
            implicit_function=implicit_function,
            gradient_function=gradient_function,
            model=model,
            location=location,
            faceIDs=faceIDs,
            vertexIDs=vertexIDs,
            penaltyParameter=penaltyParameter,
            doProximityCheck=doProximityCheck,
            proximityFactor=proximityFactor,
        )

        self._frictionCoefficient = frictionCoefficient
        self._tangentialPenaltyParameter = (
            tangentialPenaltyParameter if tangentialPenaltyParameter is not None else penaltyParameter
        )

        # Committed stick forces (state at the last converged increment) and
        # the trial values of the current increment, per contact point.
        self._frictional_forces = [np.zeros(self._fieldSize) for _ in self._constrained_points]
        self._trial_frictional_forces = [np.zeros(self._fieldSize) for _ in self._constrained_points]

    def acceptLastState(self):
        self._frictional_forces = [f.copy() for f in self._trial_frictional_forces]

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep):
        # Contact points which do not (or no longer) penetrate keep a zero
        # trial force, so separation resets the friction history on acceptance.
        self._trial_frictional_forces = [np.zeros(self._fieldSize) for _ in self._constrained_points]
        super().applyConstraint(dU_, PExt, V, timeStep)

    def _applyAdditionalForces(
        self,
        pt_idx: int,
        constrained_point: np.ndarray,
        normal: np.ndarray,
        N_vec: np.ndarray,
        deltaU_point: np.ndarray,
        g: float,
        PExt: np.ndarray,
        K_UU: np.ndarray,
    ):
        if self._frictionCoefficient <= 0.0:
            return

        fs = self._fieldSize
        k_t = self._tangentialPenaltyParameter

        # Tangential displacement increment (the implicit surface is fixed)
        P_t = np.eye(fs) - np.outer(normal, normal)
        deltaU_t = P_t @ deltaU_point

        # Elastic predictor from the committed state, projected onto the
        # current tangent plane
        f_t_n = P_t @ self._frictional_forces[pt_idx]
        f_t_trial = f_t_n - k_t * deltaU_t

        # Equivalent elastic slip conjugate to the trial force
        s_eff = -f_t_trial / k_t
        s_eff_norm = np.linalg.norm(s_eff)

        f_n_mag = self._penaltyParameter * g
        f_t_max = self._frictionCoefficient * f_n_mag

        is_slipping = (k_t * s_eff_norm >= f_t_max) and (f_t_max > 0.0) and (s_eff_norm > 1e-14)

        if s_eff_norm > 1e-14:
            t_s = s_eff / s_eff_norm
        else:
            t_s = np.zeros(fs)

        if is_slipping:
            f_t = -f_t_max * t_s
        else:
            f_t = f_t_trial

        self._trial_frictional_forces[pt_idx] = f_t

        # Assemble the friction force: following the assembly convention of
        # the base class' normal contact, the assembled contribution is the
        # negative of the physical force acting on the particle.
        n_nodes = len(self._nodes)
        for i in range(n_nodes):
            start_idx = i * fs
            PExt[start_idx : start_idx + fs] -= N_vec[i] * f_t

        # Consistent tangent (frozen geometry within the increment)
        G = np.zeros((fs, n_nodes * fs))
        for i in range(n_nodes):
            G[:, i * fs : (i + 1) * fs] = N_vec[i] * P_t

        if is_slipping:
            dg_dU = np.zeros(n_nodes * fs)
            w_t = np.zeros(n_nodes * fs)
            for i in range(n_nodes):
                dg_dU[i * fs : (i + 1) * fs] = N_vec[i] * (-normal)
                w_t[i * fs : (i + 1) * fs] = N_vec[i] * t_s

            P_slip = np.eye(fs) - np.outer(t_s, t_s)
            K_UU += self._frictionCoefficient * self._penaltyParameter * np.outer(w_t, dg_dU)
            K_UU += (f_t_max / s_eff_norm) * (G.T @ P_slip @ G)
        else:
            K_UU += k_t * (G.T @ G)


def FrictionalParticlePenaltyContactImplicitSurfaceConstraintFactory(
    baseName: str,
    implicit_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    frictionCoefficient: float = 0.0,
    tangentialPenaltyParameter: float = None,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
):
    constraints = []

    if isinstance(particleCollection, EntityBasedSurface):
        elements = particleCollection.getEntities()
    else:
        elements = particleCollection

    if location == "center":
        for i, particle in enumerate(elements):
            c = FrictionalParticlePenaltyContactImplicitSurfaceConstraint(
                name=f"{baseName}_{i}",
                particle=particle,
                implicit_function=implicit_function,
                gradient_function=gradient_function,
                model=model,
                location=location,
                penaltyParameter=penaltyParameter,
                frictionCoefficient=frictionCoefficient,
                tangentialPenaltyParameter=tangentialPenaltyParameter,
                doProximityCheck=doProximityCheck,
                proximityFactor=proximityFactor,
            )
            constraints.append(c)

    elif location == "face":
        for i, particle in enumerate(elements):
            faces = particle.geometry.getFaces() if faceIDs is None else faceIDs
            for j in faces:
                c = FrictionalParticlePenaltyContactImplicitSurfaceConstraint(
                    name=f"{baseName}_{i}_f{j}",
                    particle=particle,
                    implicit_function=implicit_function,
                    gradient_function=gradient_function,
                    model=model,
                    location=location,
                    faceIDs=j,
                    penaltyParameter=penaltyParameter,
                    frictionCoefficient=frictionCoefficient,
                    tangentialPenaltyParameter=tangentialPenaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
                constraints.append(c)

    elif location == "vertex":
        for i, particle in enumerate(elements):
            vertices = particle.geometry.getVertices() if vertexIDs is None else vertexIDs
            for j in vertices:
                c = FrictionalParticlePenaltyContactImplicitSurfaceConstraint(
                    name=f"{baseName}_{i}_v{j}",
                    particle=particle,
                    implicit_function=implicit_function,
                    gradient_function=gradient_function,
                    model=model,
                    location=location,
                    vertexIDs=j,
                    penaltyParameter=penaltyParameter,
                    frictionCoefficient=frictionCoefficient,
                    tangentialPenaltyParameter=tangentialPenaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
                constraints.append(c)

    return constraints
