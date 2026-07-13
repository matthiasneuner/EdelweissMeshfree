# -*- coding: utf-8 -*-
from collections.abc import Callable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase


class PenaltyDirichletExplicit(MPMConstraintBase):
    def __init__(
        self,
        name: str,
        model,
        nSet,
        field: str,
        values: dict,
        penaltyParameter: float,
        f_t: Callable[[float], float] = None,
    ):
        self._name = name
        self._model = model
        self.nSet = nSet
        self._nodes = list(nSet.nodes)
        self._field = field
        self._penaltyParameter = penaltyParameter
        self._domainSize = model.domainSize
        self._fieldSize = getFieldSize(field, self._domainSize)

        self._prescribedStepDelta = {int(k): float(v) for k, v in values.items()}
        # By default, use the full prescribed value in explicit simulations
        # (consistent with ParticlePenaltyWeakDirichletExplicit).
        self._f_t = f_t if f_t is not None else lambda x: 1.0

        self.penaltyForce = np.zeros(self._fieldSize)
        self.perNodePenaltyForce = np.zeros((len(self._nodes), self._fieldSize))
        self.isActive = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return [[self._field]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list:
        return []

    @property
    def active(self) -> bool:
        return self.isActive

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list):
        pass

    def updateConnectivity(self, model) -> bool:
        return False

    def applyConstraint(self, PExt: np.ndarray, timeStep):
        nodeField = self._model.nodeFields.get(self._field)
        if nodeField is None or "U" not in nodeField:
            return

        self.penaltyForce.fill(0.0)
        self.perNodePenaltyForce.fill(0.0)

        for i, prescribedComponent in self._prescribedStepDelta.items():
            P_i = PExt[i :: self._fieldSize]
            target_value = prescribedComponent * self._f_t(timeStep.stepProgress)

            total_force = 0.0

            for node_idx, node in enumerate(self._nodes):
                actual_u = nodeField.subset(node)["U"][0][i]
                deviation = actual_u - target_value
                force_scalar = -self._penaltyParameter * deviation

                # Apply force
                P_i[node_idx] += force_scalar
                total_force += force_scalar
                self.perNodePenaltyForce[node_idx, i] = force_scalar

            self.penaltyForce[i] = total_force
