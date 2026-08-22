from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyZeroDisplacementComponentExplicit(MPMConstraintBase):
    """Penalise one displacement component of a particle towards zero, for EXPLICIT simulations.

    The penalty acts on the *signed* displacement of the particle centre away from its initial
    position,

    .. math:: F = -k \\, \\left( x_n - x_n^{(0)} \\right),

    so it is two-sided and permanently active.

    This is what a symmetry plane needs, and it is why
    :class:`ParticlePenaltyCartesianBoundaryConstraintExplicit` cannot serve that purpose: the latter
    penalises only positive penetration of a fixed plane, and material adjacent to the symmetry plane
    of a symmetric impact moves *away* from it -- the one-sided wall never engages and the cut face
    ends up effectively free. Measuring against the initial position rather than the plane coordinate
    also matters, because particle centres sit half a particle off the plane and must stay there.

    The same constraint serves as a simple support when applied with the out-of-plane component.

    Parameters
    ----------
    name
        Unique name of the constraint.
    particle
        The particle to be constrained.
    component
        Index of the displacement component to be held at zero.
    model
        The model.
    penaltyParameter
        The penalty stiffness.
    """

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        component: int,
        model: MPMModel,
        penaltyParameter: float = 1e5,
    ):
        if component < 0 or component >= model.domainSize:
            raise ValueError(f"component must be in [0, {model.domainSize - 1}], got {component}")

        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._particle = particle
        self._component = component
        self._penaltyParameter = penaltyParameter
        self.reactionForce = 0.0

        self._referenceCoordinate = float(particle.getCenterCoordinates()[component])

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

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model):
        """Cache the nodes carrying this particle's kernel functions."""

        nodes = [kf.node for kf in self._particle.kernelFunctions]
        hasChanged = nodes != self._nodes
        self._nodes = nodes
        return hasChanged

    def applyConstraint(self, PExt: np.ndarray, timeStep: TimeStep):
        """Add the two-sided penalty force restoring the constrained component to zero."""

        currentPoint = self._particle.getCenterCoordinates()

        forceScalar = -self._penaltyParameter * (currentPoint[self._component] - self._referenceCoordinate)

        N = self._particle.getInterpolationVector(currentPoint).flatten()
        PExt[self._component :: self._fieldSize] += N * forceScalar

        self.reactionForce = abs(forceScalar)


def ParticleExplicitPenaltyZeroDisplacementComponentFactory(
    baseName: str,
    component: int,
    particleCollection: Iterable[BaseParticle],
    model: MPMModel,
    penaltyParameter: float = 1e5,
):
    """Create one :class:`ParticlePenaltyZeroDisplacementComponentExplicit` per particle."""

    constraints = dict()
    for i, p in enumerate(particleCollection):
        name = f"{baseName}_{i}"
        constraints[name] = ParticlePenaltyZeroDisplacementComponentExplicit(
            name, p, component, model, penaltyParameter=penaltyParameter
        )
    return constraints
