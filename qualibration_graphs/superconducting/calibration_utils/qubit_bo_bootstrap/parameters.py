"""
parameters.py — NodeParameters for the BO qubit bootstrap node (03e).

All BO-specific knobs live in ``NodeSpecificParameters``.  The final
``Parameters`` class inherits from:
  - ``NodeParameters``         (qualibrate base, adds node name / versioning)
  - ``CommonNodeParameters``   (simulate, load_data_id, timeout, reset_type, …)
  - ``NodeSpecificParameters`` (everything below)
  - ``QubitsExperimentNodeParameters`` (qubits list, multiplexed flag)
"""

from typing import Literal, Optional

from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """All tunable knobs for the GP-Bayesian-Optimisation qubit bootstrap.

    The BO explores a 2D (or 4D) search space:
      2D default: (ω_d, V_d)       — drive frequency and OPX IF amplitude
      4D opt-in:  (ω_d, V_d, ω_r, V_r) — also sweeps readout frequency/amplitude

    Three-phase strategy
    --------------------
    Phase 1a – Broad LHS:   ``n_initial_lhs`` quasi-random points over ±``freq_search_radius_mhz``
    Phase 1b – Zoom LHS:    ``n_zoom_lhs`` points in ±``zoom_radius_mhz`` around the Phase-1a best
    Phase 2  – BO (EI/UCB): up to ``n_bo_iterations`` guided acquisitions; stops early when the
                            GP-predicted optimum moves less than ``convergence_tolerance_mhz``
    """

    # ── Time-Rabi measurement ─────────────────────────────────────────────────
    num_shots: int = 100
    """Number of single-shot averages per time-Rabi trace. More shots → lower quantum
    projection noise but longer measurement time. Default 100 suits hardware BO
    (total ~26 evals × 100 shots × ~2 µs/shot × overhead ≈ a few seconds)."""

    min_duration_ns: int = 16
    """Minimum qubit drive pulse duration in ns. Must be a multiple of 4 (OPX clock).
    16 ns = 4 clock cycles is the minimum playable pulse on OPX+."""

    max_duration_ns: int = 448
    """Maximum qubit drive pulse duration in ns (exclusive). Must be a multiple of 4.
    448 ns = 9 periods of a 20 MHz target Rabi → gives enough oscillation cycles
    for the FFT-based frequency estimator to work reliably."""

    duration_step_ns: int = 4
    """Duration sweep step in ns. Must be a multiple of 4.
    Step = 4 ns (one clock cycle) gives the densest possible time grid."""

    operation: str = "x180"
    """QUAM operation name to play for the time-Rabi pulse.
    Must be defined in ``qubit.xy.operations``.  The BO will update its amplitude."""

    # ── Wolff cost function weights ───────────────────────────────────────────
    target_rabi_freq_mhz: float = 20.0
    """Target generalised Rabi frequency Ω_T in MHz.
    The cost function penalises |Ω_R − Ω_T| / Ω_T.  Set so that Ω_T × max_duration_ns
    covers the desired number of oscillation periods (e.g. 9 periods → max_duration = 450 ns)."""

    w_rabi: float = 10.0
    """Weight of the Rabi-frequency term in the Wolff cost: w_rabi × |Ω_R−Ω_T|/Ω_T.
    Dominates the landscape → drives the optimizer toward resonance first."""

    w_amp: float = 3.0
    """Weight of the oscillation-amplitude term: −w_amp × log(A).
    A = Ω_d/Ω_R peaks sharply at resonance (A→1) and falls off quickly away from it."""

    w_snr: float = 1.0
    """Weight of the SNR term: −w_snr × log(SNR).
    Penalises very low drive amplitudes and poor readout conditions."""

    # ── Drive frequency search space ─────────────────────────────────────────
    freq_search_radius_mhz: float = 200.0
    """Half-width of the broad frequency search window around the QUAM prior in MHz.
    The BO searches ω_d ∈ [prior − radius, prior + radius].
    Also clamped so the OPX IF (= ω_d − LO) stays within the OPX+ ±250 MHz bandwidth."""

    # ── Drive amplitude search space ─────────────────────────────────────────
    amp_search_min_db: float = -20.0
    """Lower bound for the drive amplitude search in dB relative to the QUAM prior.
    −20 dB = 1/10 of the prior amplitude.  Only the OPX IF voltage is swept;
    for wider ranges, adjust the Octave gain before running the BO."""

    amp_search_max_db: float = 20.0
    """Upper bound for the drive amplitude search in dB relative to the QUAM prior.
    +20 dB = 10× the prior amplitude, clipped to ``max_amplitude_opx``."""

    max_amplitude_opx: float = 0.45
    """Hard cap on the OPX IF waveform amplitude in Volts.
    OPX+ IQChannel max is 0.5 V; 0.45 V leaves a safety margin for crosstalk."""

    # ── Three-phase BO loop sizes ─────────────────────────────────────────────
    n_initial_lhs: int = 12
    """Number of quasi-random Latin Hypercube points in Phase 1a (broad scan).
    12 points over ±200 MHz → average spacing ~33 MHz.  Sufficient for the GP
    to identify the rough region containing the qubit."""

    n_zoom_lhs: int = 10
    """Number of LHS points in Phase 1b (zoom scan around best Phase-1a point).
    10 points over ±50 MHz → average spacing ~10 MHz, giving the GP enough
    resolution to distinguish the true minimum from nearby local minima."""

    zoom_radius_mhz: float = 50.0
    """Half-width of the zoom window around the best Phase-1a point in MHz.
    Should be wide enough to capture the true minimum if the Phase-1a best is
    ~10–20 MHz off.  Default 50 MHz works for priors within ±150 MHz of truth."""

    n_bo_iterations: int = 40
    """Maximum number of BO acquisition iterations in Phase 2.
    Convergence typically occurs in 3–8 iterations after the two LHS phases;
    40 is a safe upper bound that is rarely reached."""

    convergence_tolerance_mhz: float = 1.0
    """BO convergence threshold: stop if the GP-predicted optimum frequency moves
    less than this value (in MHz) between consecutive iterations."""

    acq: Literal["EI", "UCB"] = "EI"
    """Acquisition function for the BO.
    'EI'  (Expected Improvement) – balanced explore/exploit; recommended default.
    'UCB' (Upper Confidence Bound) – more aggressive exploration; useful when the
          landscape has many local minima."""

    bo_seed: Optional[int] = None
    """Random seed for the BO LHS sampler and GP initialisation.
    None = use the same seed as the QualibrationNode ``seed`` parameter."""

    # ── 4D joint readout optimisation (opt-in) ───────────────────────────────
    optimize_readout_jointly: bool = False
    """If True, expand the search to 4D: also optimise (ω_r, V_r) simultaneously.
    Off by default — resonator bringup already provides a reliable readout prior,
    so joint optimisation is only useful when readout fidelity is known to be poor.
    Enabling this requires more LHS points (n_initial_lhs ≥ 30, n_zoom_lhs ≥ 30)."""

    readout_freq_radius_mhz: float = 3.0
    """Half-width of the readout frequency search window in MHz (4D only).
    Kept tight (±3 MHz) because the resonator bringup prior is reliable."""

    readout_amp_search_min_db: float = -6.0
    """Lower bound for the readout amplitude search in dB relative to QUAM prior (4D only)."""

    readout_amp_search_max_db: float = 6.0
    """Upper bound for the readout amplitude search in dB relative to QUAM prior (4D only)."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Full parameter set for 03e_qubit_bo_bootstrap.

    Inherits from:
      NodeParameters                 – qualibrate versioning / node identity
      CommonNodeParameters           – simulate, load_data_id, timeout, reset_type,
                                       num_shots (overridden above), use_state_discrimination
      NodeSpecificParameters         – all BO-specific knobs (see above)
      QubitsExperimentNodeParameters – qubits: List[str], multiplexed: bool
    """
