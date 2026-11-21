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
#  ---------------------------------------------------------------------
"""A wave through the ties between particles and finite elements, in explicit dynamics: a bar of finite elements with a
segment of particles in its middle, tied to both of its parts, carries a pulse from one end to the other, and the ties
must let it through as the bar of finite elements alone does.

.. code-block:: text

      pulse ->  +--------------------+=============+--------------------+
                |  finite elements   |  particles  |  finite elements   |   free end
                +--------------------+=============+--------------------+
                     ^ probe        tie           tie       ^ probe

The bar, ``WIDTH`` x ``WIDTH`` in section and elastic (compressible neo-Hookean, Poisson's ratio zero, so that the wave
is one-dimensional), is of ``C3D8UL`` elements and of displacement particles (``DisplacementSQCNIxNSNI/3D/Hexa``), all
of the size ``SIZE``; the ties are EdelweissFE's one contact constraint with the law ``type=tied``
(:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`), the quadrature points of the faces of the particles
tied to the faces of the elements. The end of the bar is moved by ``U0 (1 - cos(pi t / T)) / 2`` and then held: a pulse
of the velocity ``v = U0 pi / (2 T) sin(pi t / T)``, ``c T`` long. EdelweissFE's ``NEDParallel`` integrates it.

A tie of the stiffness ``k`` per unit area between two equal bars of the impedance ``Z = rho c`` reflects a harmonic
wave of the circular frequency ``omega`` with the amplitude ``R = omega Z / sqrt((omega Z)^2 + 4 k^2)``, about
``omega Z / (2 k)``: a soft tie reflects, a stiff one lets the wave through. Its stiffness is limited by the time step,
though: a spring as stiff as the material over the size of a particle, ``E / h``, is as stiff as a particle itself.
The time increment is therefore set here (the Courant number) for the stiffness of the tie.

What it checks: the pulse that a bar with a tied segment reflects at the probe in front of the first tie, and the
pulse it transmits to the probe behind the second one, compared with the bar of finite elements alone. A segment of
elements, tied in, reflects what the compliance of its two ties does, and nothing else. A segment of particles reflects
a little more: the particles carry a wave a little differently from the elements (their own dispersion), and that
mismatch, not the ties, sets what is reflected and how the transmitted pulse changes; both shrink as the particles and
elements get smaller.
"""

import numpy as np
import pytest
from edelweissfe.config.configurator import loadConfiguration
from edelweissfe.constraints.base.contactpointsonslavesurface import (
    ContactPointsOnSlaveSurface,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    Constraint as SurfaceContact,
)
from edelweissfe.constraints.surfacetodeformablesurfacepenalty import (
    SurfaceToDeformableSurfacePenaltySchema,
)
from edelweissfe.generators.boxgen import BoxgenSchema
from edelweissfe.generators.boxgen import Generator as BoxGenerator
from edelweissfe.generators.surfaceelementgenerator import buildContactFacets
from edelweissfe.journal.journal import Journal
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sections.solid import Section as SolidSection
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.boxhexaparticlegridgenerator import (
    generateBoxHexaParticleGrid,
)
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

#: The bar: WIDTH x WIDTH in section, finite elements from 0 to START, particles from START to END, finite elements
#: from END to LENGTH; elements and particles of SIZE [mm].
WIDTH, START, END, LENGTH, SIZE = 10.0, 100.0, 150.0, 250.0, 5.0
#: Young's modulus [MPa], density [t/mm^3], and the speed of the wave [mm/s].
E, RHO = 30000.0, 2.4e-9
C = np.sqrt(E / RHO)
#: The pulse: the end moves by U0 [mm] over the time T [s], so the pulse is 50 mm long.
U0, T = 1e-3, 50.0 / C
#: The probes, in front of the first tie and behind the second one.
PROBES = (60.0, 200.0)
#: The time the test runs: until the pulse reflected at the free end comes back to the second probe.
END_TIME = (2 * LENGTH - PROBES[1]) / C
#: The stiffness of the ties per unit area, in units of E / SIZE, and the Courant number that keeps them stable.
STIFFNESS, COURANT_NUMBER = 8.0, 0.3

MATERIAL = {"name": "CompressibleNeoHooke", "properties": np.array([E / 3, E / 2, RHO])}


def addFiniteElementBar(model: MPMModel, journal: Journal, name: str, z0: float, length: float):
    """A bar of ``C3D8UL`` elements from ``z0``, ``length`` long, with its section."""

    with model.topology.changes():
        BoxGenerator(
            name,
            model,
            journal,
            configuration=BoxgenSchema(
                x0=0.0,
                y0=0.0,
                z0=z0,
                lX=WIDTH,
                lY=WIDTH,
                lZ=length,
                nX=round(WIDTH / SIZE),
                nY=round(WIDTH / SIZE),
                nZ=round(length / SIZE),
                elType="C3D8UL",
            ),
        )
    model.sections[name] = SolidSection(name, model, MATERIAL, [model.elementSets[f"{name}_all"]])


def addParticleSegment(model: MPMModel, journal: Journal):
    """The segment of displacement particles, on kernel functions matched to them."""

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)
    material = {"material": MATERIAL["name"], "properties": MATERIAL["properties"]}
    generateBoxHexaParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "DisplacementSQCNIxNSNI/3D/Hexa", number, vertices, 0.0, approximation, material
        ),
        name="segment",
        x0=0.0,
        y0=0.0,
        z0=START,
        l=WIDTH,
        h=WIDTH,
        t=END - START,
        nX=round(WIDTH / SIZE),
        nY=round(WIDTH / SIZE),
        nZ=round((END - START) / SIZE),
    )
    generateKernelMatchingToParticle(
        model,
        journal,
        lambda node, supportSize: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3
        ),
        model.particleSets["segment_all"],
        supportScalingFactor=2.4,
        firstKernelFunctionNumber=max(model.nodes) + 1,
    )
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    model.particleKernelDomains["segment"] = domain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(domain, 3, journal, bondParticlesToKernelFunctions=True)
    )


def facets(model: MPMModel, journal: Journal, surface: str) -> str:
    """The contact facets of a surface of finite elements, ``<surface>_facets``."""

    with model.topology.changes():
        return buildContactFacets(model, surface, surface, "midside", "facetConsistent", journal)[0]


def tieTheSegment(model: MPMModel, journal: Journal, segment: str):
    """Tie the ends of the segment to the faces of the elements they meet: the quadrature points of the faces of its
    particles, or of its elements."""

    # boxgen names the faces of elements back (z min) and front (z max)
    masters = [facets(model, journal, face) for face in ("left_front", "right_back")]
    if segment == "particles":
        ends = [ContactPointsOnParticleFaces(model.surfaces[f"segment_{end}"], nQuadraturePoints=4) for end in ("bottom", "top")]  # fmt: skip
    else:
        ends = [ContactPointsOnSlaveSurface(model.elementSets[facets(model, journal, f"segment_{end}")], 3, end) for end in ("back", "front")]  # fmt: skip
    configuration = SurfaceToDeformableSurfacePenaltySchema(
        penalty=STIFFNESS * E / SIZE, contactType="tied", searchDistance=0.1 * SIZE
    )
    for name, points, master in zip(("first", "second"), ends, masters):
        model.constraints[name] = SurfaceContact(
            name, model, points, model.elementSets[master], journal, configuration=configuration
        )


class VelocityAtTheProbes(OutputManagerBase):
    """Records, in every increment, the mean axial velocity of the nodes of the elements in the cross-sections of the
    probes."""

    checkpointedState = {}

    def __init__(self, model: MPMModel):
        self.name = "velocityAtTheProbes"
        self._model = model
        displacement = model.nodeFields["displacement"]
        nodes = {node for element in model.elements.values() for node in element.nodes}
        self._probes = [
            [displacement.indexOfNode(node) for node in nodes if abs(node.coordinates[2] - z) < 1e-9] for z in PROBES
        ]
        self.times, self.velocities = [], []

    def initializeJob(self):
        pass

    def initializeStep(self, step):
        pass

    def finalizeIncrement(self, **kwargs):
        V = self._model.nodeFields["displacement"]["V"]
        self.times.append(self._model.time)
        self.velocities.append([V[probe, 2].mean() for probe in self._probes])

    def finalizeFailedIncrement(self, **kwargs):
        pass

    def finalizeStep(self):
        pass

    def finalizeJob(self):
        pass


def run(bar: str) -> tuple[np.ndarray, np.ndarray]:
    """Send the pulse through the bar.

    Parameters
    ----------
    bar
        "finite elements" (the bar of elements alone), "particles" (a segment of particles tied in its middle), or
        "tied finite elements" (a segment of elements tied in its middle).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The times, and the velocities at the two probes.
    """

    journal = Journal(verbose=False)
    model = MPMModel(3)
    if bar in ("particles", "tied finite elements"):
        addFiniteElementBar(model, journal, "left", 0.0, START)
        addFiniteElementBar(model, journal, "right", END, LENGTH - END)
        if bar == "particles":
            addParticleSegment(model, journal)
            tieTheSegment(model, journal, "particles")
        else:
            addFiniteElementBar(model, journal, "segment", START, END - START)
            tieTheSegment(model, journal, "finite elements")
        end = model.nodeSets["left_back"]
    else:
        addFiniteElementBar(model, journal, "bar", 0.0, LENGTH)
        end = model.nodeSets["bar_back"]
    model.prepareYourself(journal)

    stepActions = StepActionCollection()
    pulse = lambda progress: 0.5 * (1.0 - np.cos(np.pi * min(progress * END_TIME / T, 1.0)))  # noqa: E731
    stepActions["dirichlet"]["pulse"] = Dirichlet("pulse", end, "displacement", {2: U0}, model, journal, f_t=pulse)

    jobInfo = loadConfiguration({})
    solver = NEDParallel(jobInfo, journal, **{"courant-number": COURANT_NUMBER, "output-frequency": 1})
    probes = VelocityAtTheProbes(model)
    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    step = AdaptiveStepForExplicitSimulations(
        0,
        model,
        fieldOutputController,
        journal,
        jobInfo,
        solver,
        [probes],
        stepActions,
        stepLength=END_TIME,
        startInc=1.0,
        maxInc=1.0,
        minInc=1e-12,
        maxNumInc=10**5,
    )
    step.solve()
    fieldOutputController.finalizeJob()
    return np.array(probes.times), np.array(probes.velocities)


def reflectedAndTransmitted(bar: str, reference: tuple[np.ndarray, np.ndarray]) -> dict:
    """The pulses of a bar with a tied segment that differ from those of the bar of elements alone, at its time
    increments: the energy of the reflected pulse (at the first probe, after the incident one has passed) and of the
    difference of the transmitted pulse (at the second probe), each relative to the energy of the incident pulse, which
    is proportional to the integral of the squared velocity.

    Parameters
    ----------
    bar
        The bar with a tied segment, see :func:`run`.
    reference
        What :func:`run` returns for the bar of finite elements alone.

    Returns
    -------
    dict
        The energies, and the largest velocity at the probes, of this bar and of the bar of elements alone.
    """

    times, velocities = run(bar)
    referenceTimes, referenceVelocities = reference
    reference = np.column_stack([np.interp(times, referenceTimes, v) for v in referenceVelocities.T])
    difference = velocities - reference

    def energy(v: np.ndarray, window: np.ndarray) -> float:
        return float(np.trapezoid(v[window] ** 2, times[window]))

    incident = energy(reference[:, 0], times < (PROBES[0] / C + T) * 1.05)
    afterTheIncident = times > (PROBES[0] / C + T) * 1.05
    return {
        "reflected": energy(difference[:, 0], afterTheIncident) / incident,
        "transmitted": energy(velocities[:, 1], times > 0) / incident,
        "differenceOfTheTransmitted": energy(difference[:, 1], times > 0) / incident,
        "largestVelocity": float(np.abs(velocities).max()),
        "largestVelocityOfTheBarAlone": float(np.abs(referenceVelocities).max()),
    }


@pytest.fixture(autouse=True)
def _changeTestDir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def reflectionOfATie() -> float:
    """The energy a single tie reflects of a harmonic wave at the dominant circular frequency of the pulse, ``pi / T``,
    relative to the incident one: ``R^2``."""

    omegaZ = np.pi / T * RHO * C
    return omegaZ**2 / (omegaZ**2 + 4 * (STIFFNESS * E / SIZE) ** 2)


def test_ties_between_finite_elements_reflect_what_their_compliance_does():
    """A segment of elements tied into the bar: the two ties reflect about what two springs of their stiffness between
    equal bars reflect, and let the rest of the pulse through unchanged."""

    result = reflectedAndTransmitted("tied finite elements", run("finite elements"))
    assert result["largestVelocity"] < 1.05 * result["largestVelocityOfTheBarAlone"]  # stable
    assert 0.5 * reflectionOfATie() < result["reflected"] < 4 * reflectionOfATie()
    assert result["transmitted"] == pytest.approx(1.0, abs=4 * reflectionOfATie())
    assert result["differenceOfTheTransmitted"] < 5 * reflectionOfATie()


def test_ties_between_particles_and_finite_elements_let_the_pulse_through():
    """A segment of particles tied into the bar: it reflects a small part of the pulse, more than the ties alone (the
    particles carry a wave a little differently from the elements), and transmits nearly all of it; both differences
    shrink with the size of the particles and elements."""

    reference = run("finite elements")
    result = reflectedAndTransmitted("particles", reference)
    assert result["largestVelocity"] < 1.05 * result["largestVelocityOfTheBarAlone"]  # stable
    assert result["reflected"] < 0.01
    assert result["transmitted"] == pytest.approx(1.0, abs=0.03)
    assert result["differenceOfTheTransmitted"] < 0.06


if __name__ == "__main__":
    reference = run("finite elements")
    print(f"one tie reflects {reflectionOfATie():.3e}")
    for bar in ("tied finite elements", "particles"):
        print(bar, reflectedAndTransmitted(bar, reference))
