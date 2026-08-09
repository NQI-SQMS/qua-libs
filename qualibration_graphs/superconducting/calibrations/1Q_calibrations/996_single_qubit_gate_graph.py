"""
Simple single-qubit-gate tune-up graph (flux-tunable transmons).

Scope: bring the single-qubit x180/x90 gates up — assuming the readout already works
(run 995_readout_chain_graph.py first). Like 995 it demonstrates a **sub-graph** plus a
**conditional loop**, and nothing more (no failure routing / sink).

Workflow:
    qubit_spectroscopy            (find the qubit 0->1 frequency)
      -> qubit_spectroscopy_vs_flux   (track it vs flux -> idle point)
      -> power_rabi                   (coarse pi-pulse amplitude)
      -> [pi_pulse_refine]            <-- sub-graph, looped:
             ramsey                   (refine the frequency; writes the detuning)
               -> power_rabi_error_amplification_x180   (refine x180 amplitude)
               -> power_rabi_error_amplification_x90    (refine x90 amplitude)
               -> DRAG_calibration                      (DRAG coefficient)
The `pi_pulse_refine` sub-graph repeats (conditional loop) until the residual Ramsey
detuning measured at the top of an iteration is within `detuning_tolerance_hz` (the loop
re-runs the whole sub-graph, so each iteration's Ramsey reflects the previous round's
corrections), or `max_refine_iterations` is hit.

See 998_adaptive_graph_tutorial.ipynb for the building blocks and 999_adaptive_graph.py
for the full adaptive RB+retune graph.
"""

# %%
from typing import List, Optional

from qualibrate import GraphParameters, QualibrationGraph, QualibrationLibrary, QualibrationNode
from qualibrate.core.orchestration.basic_orchestrator import BasicOrchestrator

library = QualibrationLibrary.get_active_library()


# %% {Graph parameters}
class Parameters(GraphParameters):
    qubits: Optional[List[str]] = None
    """Qubits to operate on. None = all active qubits."""

    detuning_tolerance_hz: float = 50_000.0
    """Stop refining once |Ramsey frequency offset| is below this (Hz)."""

    max_refine_iterations: int = 3
    """Hard cap on how many times the pi-pulse-refine sub-graph repeats."""


parameters = Parameters()


class SubgraphParameters(GraphParameters):
    qubits: Optional[List[str]] = None


# %% {Loop condition: is the qubit frequency still drifting?}
def detuning_not_converged(subgraph: QualibrationGraph, target: str) -> bool:
    """Per-qubit loop predicate: True (keep looping) while the Ramsey detuning is still
    larger than the tolerance (or the fit failed)."""
    ramsey_node = subgraph._elements["ramsey"]
    fit = ramsey_node.results.get("fit_results", {}).get(target, {})
    if not fit or not fit.get("success"):
        return True  # no/failed fit -> retry
    return abs(float(fit.get("freq_offset", 1e12))) > parameters.detuning_tolerance_hz


# %% {Build the graph}
with QualibrationGraph.build("single_qubit_gate_tuneup", parameters=parameters) as graph:
    # ---- 1. Find the qubit frequency and a coarse pi-pulse ----
    qubit_spectroscopy = library.nodes["08_qubit_spectroscopy"].copy(name="qubit_spectroscopy")
    graph.add_node(qubit_spectroscopy)

    qubit_spectroscopy_vs_flux = library.nodes["09_qubit_spectroscopy_vs_flux"].copy(
        name="qubit_spectroscopy_vs_flux"
    )
    graph.add_node(qubit_spectroscopy_vs_flux)
    graph.connect(qubit_spectroscopy, qubit_spectroscopy_vs_flux)

    power_rabi = library.nodes["11_power_rabi"].copy(name="power_rabi")
    graph.add_node(power_rabi)
    graph.connect(qubit_spectroscopy_vs_flux, power_rabi)

    # ---- 2. Pi-pulse refine sub-graph (frequency -> amplitudes -> DRAG) ----
    with QualibrationGraph.build(
        "pi_pulse_refine",
        parameters=SubgraphParameters(),
        orchestrator=BasicOrchestrator(skip_failed=False),
    ) as pi_pulse_refine:
        # `ramsey` is the metric node read by the loop predicate; it heads the sub-graph
        # so each looped iteration re-measures the residual detuning.
        ramsey = library.nodes["12_ramsey"].copy(
            name="ramsey",
            frequency_detuning_in_mhz=0.1,
            log_or_linear_sweep="linear",
        )
        power_rabi_error_amplification_x180 = library.nodes["11_power_rabi"].copy(
            name="power_rabi_error_amplification_x180",
            max_number_pulses_per_sweep=100,
            min_amp_factor=0.95,
            max_amp_factor=1.05,
            amp_factor_step=0.005,
        )
        power_rabi_error_amplification_x90 = library.nodes["11_power_rabi"].copy(
            name="power_rabi_error_amplification_x90",
            max_number_pulses_per_sweep=100,
            min_amp_factor=0.95,
            max_amp_factor=1.05,
            amp_factor_step=0.005,
            operation="x90",
            update_x90=False,
        )
        DRAG_calibration = library.nodes["13_drag_calibration_180_minus_180"].copy(
            name="DRAG_calibration"
        )

        pi_pulse_refine.add_node(ramsey)
        pi_pulse_refine.add_node(power_rabi_error_amplification_x180)
        pi_pulse_refine.add_node(power_rabi_error_amplification_x90)
        pi_pulse_refine.add_node(DRAG_calibration)
        pi_pulse_refine.connect(ramsey, power_rabi_error_amplification_x180)
        pi_pulse_refine.connect(power_rabi_error_amplification_x180, power_rabi_error_amplification_x90)
        pi_pulse_refine.connect(power_rabi_error_amplification_x90, DRAG_calibration)

    graph.add_node(pi_pulse_refine)
    graph.connect(power_rabi, pi_pulse_refine)

    # ---- 3. Loop the refine sub-graph until the detuning converges ----
    graph.loop(
        pi_pulse_refine,
        on=detuning_not_converged,
        max_iterations=parameters.max_refine_iterations,
    )


# %% {Run}
if __name__ == "__main__":
    graph.run()

# %%
