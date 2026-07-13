# -*- coding: utf-8 -*-
import numpy as np
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.discreterigidbodypenaltycontact import (
    DiscreteRigidBodyPenaltyContact,
)


class FrictionalDiscreteRigidBodyPenaltyContact(DiscreteRigidBodyPenaltyContact):
    """
    Frictional penalty contact constraint for discrete rigid bodies in implicit analyses.

    Mathematical Formulation
    ------------------------
    1. Tangential Kinematics:
       The relative tangential slip increment vector Δu_t is:
         Δu_t = P_t · (Δu_p - Δu_rp - Δθ_rp × r_s)
       where:
         - P_t = I - n_s ⊗ n_s is the projection tensor onto the tangent plane.

    2. Frictional Forces (elastic-predictor / radial-return with history):
       The friction force is history dependent. The stick force committed at
       the end of the previous increment, f_t^n, is stored per particle and
       projected onto the current tangent plane. The trial force is
         f_t^trial = P_t · f_t^n - k_t · Δu_t
       with the regularized tangential contact stiffness k_t. With the
       Coulomb limit f_t^max = μ · f_n:
         - Stick (||f_t^trial|| < f_t^max):  f_t = f_t^trial
         - Slip  (||f_t^trial|| >= f_t^max): f_t = -f_t^max · t_s,
           t_s = s_eff / ||s_eff||,  s_eff = -f_t^trial / k_t
       The trial state is committed as the new history upon increment
       acceptance (:meth:`acceptLastState`); intermediate Newton iterations
       never mutate the committed state, so the result is independent of the
       iteration count.

    3. Tangent Stiffness Contribution (consistent with the frozen-geometry
       residual, see the base class):
       - Sticking:
         K_stick = k_t · (G_t^T · G_t)
         where G_t = P_t · [ N_i · I,  -I,  r_s_hat ] is the slip gradient matrix.
       - Slipping:
         K_slip = μ · k_n · (w_t ⊗ w) + (μ · f_n / ||s_eff||) · G_t^T · (I - t_s ⊗ t_s) · G_t
         where w_t = [ N_i · t_s,  -t_s,  t_s × r_s ] and w is the normal
         contact gradient from the base class.
    """

    def __init__(self, *args, frictionCoefficient: float = 0.3, viscousRegularization: float = 1e6, **kwargs):
        super().__init__(*args, **kwargs)
        self.frictionCoefficient = frictionCoefficient
        self.tangentStiffness = viscousRegularization

        self._totalNormalForce = np.zeros(self._domainSize)
        self._totalFrictionForce = np.zeros(self._domainSize)

        # Committed (converged) stick forces per particle, and the trial
        # values of the current increment.
        self._stickForces = {}
        self._trialStickForces = {}

    @property
    def totalNormalForce(self) -> np.ndarray:
        return self._totalNormalForce

    @property
    def totalFrictionForce(self) -> np.ndarray:
        return self._totalFrictionForce

    def acceptLastState(self):
        self._stickForces = dict(self._trialStickForces)

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, K, timeStep: TimeStep):
        self._trialStickForces = {}

        if not self.isActive:
            self._totalNormalForce = np.zeros(self._domainSize)
            self._totalFrictionForce = np.zeros(self._domainSize)
            return

        K = self.shapeVIJContribution(K)

        self.reactionForce = 0.0
        nDim = self._domainSize
        nRot = 3

        self._totalNormalForce = np.zeros(nDim)
        self._totalFrictionForce = np.zeros(nDim)

        node_local_dofs = self._getLocalDofMapping()
        rp_dofs = node_local_dofs[self.rigidBody.rpNode]

        # RP trial increments at current iteration
        delta_u_rp = dU[rp_dofs[0:nDim]]
        delta_theta_rp = dU[rp_dofs[nDim : nDim + nRot]]

        # Cached start-of-increment surface query and RP position
        coords = self._candidateCoords
        dists = self._candidateDists
        normals = self._candidateNormals
        rp_current = self._rpCurrent

        for idx, p in enumerate(self._candidates):
            d0 = dists[idx]
            if d0 >= self.proximityFactor:
                continue

            n_s = normals[idx]
            x_s0 = coords[idx] - d0 * n_s
            r_s = x_s0 - rp_current

            # Particle shape functions and background DOFs
            N_vec = p.getInterpolationVector(coords[idx]).flatten()
            p_nodes = [kf.node for kf in p.kernelFunctions]
            p_dofs = []
            for node in p_nodes:
                p_dofs.extend(node_local_dofs[node])

            delta_u_p = np.zeros(nDim)
            for i, node in enumerate(p_nodes):
                local_node_dofs = node_local_dofs[node]
                delta_u_p += N_vec[i] * dU[local_node_dofs]

            # 1. Normal Contact
            c = -n_s
            delta_u_rel = delta_u_p - delta_u_rp - np.cross(delta_theta_rp, r_s)
            g = -d0 + np.dot(c, delta_u_rel)

            if g < 0:
                continue

            # Normal gradient
            dg_dU = np.zeros(self.nDof)
            for i in range(len(p_nodes)):
                dg_dU[p_dofs[i * nDim : (i + 1) * nDim]] = N_vec[i] * c
            dg_dU[rp_dofs[0:nDim]] = -c
            # d g / d Δθ = r_s x n_s (see base class)
            dg_dU[rp_dofs[nDim : nDim + nRot]] = np.cross(r_s, n_s)

            f_n_mag = self._penaltyParameter * g
            self.reactionForce += f_n_mag
            self._totalNormalForce += f_n_mag * n_s

            # 2. Tangential Frictional Contact (elastic predictor / radial return)
            P_t = np.eye(nDim) - np.outer(n_s, n_s)
            delta_u_t = P_t @ delta_u_rel

            k_t = self.tangentStiffness
            f_t_n = P_t @ self._stickForces.get(p, np.zeros(nDim))
            f_t_trial = f_t_n - k_t * delta_u_t

            # Equivalent elastic slip conjugate to the trial force
            s_eff = -f_t_trial / k_t
            s_eff_norm = np.linalg.norm(s_eff)

            f_t_trial_norm = k_t * s_eff_norm
            f_t_mag_max = self.frictionCoefficient * f_n_mag

            is_slipping = (f_t_trial_norm >= f_t_mag_max) and (f_t_mag_max > 0.0) and (s_eff_norm > 1e-14)

            if s_eff_norm > 1e-14:
                t_s = s_eff / s_eff_norm
            else:
                t_s = np.zeros(nDim)

            if is_slipping:
                f_t = -f_t_mag_max * t_s
            else:
                f_t = f_t_trial

            self._trialStickForces[p] = f_t
            self._totalFrictionForce += f_t

            # 3. Assemble residual vectors
            f_contact = f_n_mag * n_s + f_t

            for i in range(len(p_nodes)):
                PExt[p_dofs[i * nDim : (i + 1) * nDim]] -= N_vec[i] * f_contact
            PExt[rp_dofs[0:nDim]] += f_contact
            # Assembled conjugate of the physical RP torque r_s x (-f_contact)
            PExt[rp_dofs[nDim : nDim + nRot]] += np.cross(r_s, f_contact)

            # 4. Tangent Stiffness (consistent with the frozen-geometry residual)
            K_mat = self._penaltyParameter * np.outer(dg_dU, dg_dU)

            r_s_hat = np.array([[0.0, -r_s[2], r_s[1]], [r_s[2], 0.0, -r_s[0]], [-r_s[1], r_s[0], 0.0]])

            # Frictional Stiffness
            K_fric = np.zeros((self.nDof, self.nDof))

            G_t = np.zeros((nDim, self.nDof))
            for i in range(len(p_nodes)):
                G_t[:, p_dofs[i * nDim : (i + 1) * nDim]] = N_vec[i] * P_t
            G_t[:, rp_dofs[0:nDim]] = -P_t
            # d Δu_rel / d Δθ = +r_s_hat (from -Δθ x r_s = +r_s_hat Δθ)
            G_t[:, rp_dofs[nDim : nDim + nRot]] = P_t @ r_s_hat

            if is_slipping:
                w_t = np.zeros(self.nDof)
                for i in range(len(p_nodes)):
                    w_t[p_dofs[i * nDim : (i + 1) * nDim]] = N_vec[i] * t_s
                w_t[rp_dofs[0:nDim]] = -t_s
                w_t[rp_dofs[nDim : nDim + nRot]] = np.cross(t_s, r_s)

                K_fric += self.frictionCoefficient * self._penaltyParameter * np.outer(w_t, dg_dU)

                P_slip = np.eye(nDim) - np.outer(t_s, t_s)
                K_fric += (f_t_mag_max / s_eff_norm) * (G_t.T @ P_slip @ G_t)
            else:
                K_fric += k_t * (G_t.T @ G_t)

            K_total = K_mat + K_fric
            K.K_pp[idx] += K_total[np.ix_(p_dofs, p_dofs)]
            K.K_prp[idx] += K_total[np.ix_(p_dofs, rp_dofs)]
            K.K_rpp[idx] += K_total[np.ix_(rp_dofs, p_dofs)]
            K.K_rprp += K_total[np.ix_(rp_dofs, rp_dofs)]
