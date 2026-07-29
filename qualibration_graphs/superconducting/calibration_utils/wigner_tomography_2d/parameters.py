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
    use_displaced_threshold: bool = False
    """When True and use_state_discrimination is True, use pair.ge_iq_threshold_displaced
    instead of the vacuum readout threshold (calibrated by node 26j)."""

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

    fock_prep_protocol: Literal["sfo", "sfp", "sfp_pf"] = "sfo"
    """Fock state preparation protocol (applies only when fock_prep_method='sideband'):

    'sfo'     — plain sideband ladder, no mid-sequence correction (default, existing
                behaviour).  No extra calibration required beyond sideband bringup.

    'sfp'     — Sideband Fock Preparation with Feedforward: after each sideband step,
                measure the qubit and retry if the transfer failed.  Requires node 37
                (qubit_gef_thresholds) to calibrate 3-state IQ thresholds first.

    'sfp_pf'  — SFP + Parity Filter: after the SFP ladder, a real-time parity Ramsey
                verifies the prepared Fock parity; the program retries up to
                pf_max_retries times before accepting the shot.  Requires node 37.
    """

    sfp_ff_repeat: int = 1
    """[sfp / sfp_pf] Consecutive successful feedforward measurements required at each
    sideband step before advancing.  1 = a single success suffices; increase to 2–3
    for higher confidence at the cost of longer preparation time."""

    sfp_max_retries: int = 20
    """[sfp / sfp_pf] Hard cutoff on feedforward attempts per sideband step.
    If this many GEF measurements pass without achieving ff_repeat consecutive
    successes, the loop exits and preparation continues to the next step.
    Prevents the program from blocking indefinitely when readout or sideband
    calibration is imperfect.  Default 20 is ~10× the expected number of
    retries for a well-calibrated system."""

    pf_max_retries: int = 30
    """[sfp_pf only] Maximum parity-filter retry attempts per shot.  If the loop
    exhausts this count without passing, the shot proceeds with an improperly prepared
    state, contributing as noise.  Increase for high-Fock targets where fidelity is
    lower and more retries are expected."""

    use_active_gef_qubit_reset: bool = True
    """[sfp / sfp_pf only] Use qubit.reset_qubit_active_gef() instead of the
    standard qubit.reset() between shots.  This is strongly recommended for SFP
    because sideband operations can leave the qubit in |f⟩; a standard g/e active
    reset cannot handle leakage and may flip |f⟩ → |e⟩ rather than |g⟩.
    Set False only if gef_centers has not been calibrated (node 15)."""

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
