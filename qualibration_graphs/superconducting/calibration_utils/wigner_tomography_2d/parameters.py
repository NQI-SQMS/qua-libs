from typing import List, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    mode_name: str = "alice"
    """Which cavity mode to probe ('alice' or 'bob')."""

    num_shots: int = 400
    """Number of averages per probe point."""

    n_fock_cutoff: int = 4
    """Hilbert-space truncation N_ph used for state reconstruction.
    The density matrix will be (N_ph × N_ph). Set to the highest
    relevant photon number + a few extra levels."""

    n_probe_points: Optional[int] = None
    """Number of optimised probe displacements (default: N_ph² + 30)."""

    probe_displacements_path: Optional[str] = None
    """Path to a pre-computed .npz file produced by a previous run.
    If the file exists the optimisation is skipped and the stored
    displacements are loaded (must have been computed for the same
    n_fock_cutoff). If None or the file does not exist, the
    optimisation runs and the result is saved there (or to a default
    location inside the node results directory). Copy the path from
    node.results['probe_displacements_path'] after the first run to
    reuse on subsequent runs."""

    fock_prep_method: Literal["sideband", "snap_displacement"] = "sideband"
    """Fock state preparation method.

    'sideband': ladder of f{j}g{j+1} sideband pi-pulses
      (requires calibrated sideband transitions in QuAM).

    'snap_displacement': sequential Displacement–SNAP protocol
      (requires snap_displacement_photons to be set, and a
       'selective_x180' operation on the qubit xy element)."""

    target_fock_level: int = 0
    """Cavity Fock state |n⟩ to prepare before tomography.
    0 → vacuum; requires increasing sideband calibrations for n > 1."""

    snap_displacement_photons: Optional[List[float]] = None
    """[snap_displacement only] Displacement amplitudes in photon units
    for each step of the D-SNAP sequence:
      [α₀, α₁, …, αₙ]   (length = target_fock_level + 1)

    The sequence applied is:
      D(α₀) → SNAP|0⟩ → D(α₁) → SNAP|1⟩ → … → D(αₙ)

    Obtain these values from prior D-SNAP calibration or from
    published optimal values for your system."""

    parity_time_ns: Optional[int] = None
    """Parity measurement wait time [ns], rounded to nearest 4 ns.
    If None, computed from chi_hz: t = 1 / (4 · |chi_hz|).
    Prefer setting this directly from node 28 results."""

    chi_hz: Optional[float] = None
    """Per-photon dispersive shift [Hz]. Used to compute parity_time_ns
    when parity_time_ns is None. Falls back to pair.chi from QuAM
    when also None."""

    use_state_discrimination: bool = True
    """True → use qubit threshold for 0/1 classification.
    False → use raw I quadrature (not recommended for Wigner tomography)."""

    use_confusion_matrix_correction: bool = False
    """Apply ge readout confusion matrix correction to averaged
    excitation probabilities before computing parity."""

    wigner_range: float = 0.0
    """Half-range of the Wigner function plot in photon units.
    0 → auto-scaled from sqrt(N_ph) + 1.5."""

    n_grid: int = 101
    """Grid resolution for the reconstructed Wigner function heatmap."""

    cavity_reset_type: Literal["thermal", "active_sideband"] = "thermal"
    """How to reset the cavity between shots.
    'thermal'         - wait thermalization_time_factor × T1.
    'active_sideband' - cascade sideband pi-pulses to actively remove photons."""

    cavity_active_cooling_fock_n: int = 1
    """Starting Fock level for active sideband cavity reset (only used when
    cavity_reset_type='active_sideband')."""

    sideband_pulse_duration_ns: Optional[int] = None
    """Override sideband pulse flat-top duration [ns] for active cavity reset.
    When None, uses the calibrated pi_flat_top_length_ns from pair.transitions."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
