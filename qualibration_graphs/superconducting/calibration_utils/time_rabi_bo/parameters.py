"""
Parameters for the 03e_qubit_bo_bootstrap QualibrationNode.

All parameters have sensible defaults derived from Wolff et al. (APS 2026)
and practical experience with SQMS fixed-frequency transmons.
"""

from typing import Literal
from qualibrate import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class TimeRabiBoParameters(NodeParameters, CommonNodeParameters, QubitsExperimentNodeParameters):
    """
    Full parameter set for the BO bootstrap node.

    Organised into sections:
      - Time-Rabi measurement settings
      - Search bounds (relative to QUAM priors)
      - BO hyperparameters
      - Cost function weights
      - Readout settings
    """

    # ── Time-Rabi measurement ──────────────────────────────────────────────────
    target_rabi_freq_mhz: float = 20.0
    """Target Rabi frequency Ω_T in MHz.

    Controls the cost function's resonance term and sets the time axis of
    every measurement (t_max = num_periods / Ω_T).

    Practical constraints:
    - Must be << 1/t_pulse_rise  to avoid bandwidth effects
    - Must be >> 1/T2*           so oscillations are visible
    - Wolff et al. use Ω_T ∈ [5, 40] MHz; 20 MHz is a safe midpoint for
      typical fixed-frequency transmons with T2* ~ 5–50 µs.
    - Higher Ω_T → shorter total pulse → less T2* decay → cleaner fit.
    - Lower  Ω_T → longer total pulse → more aliasing risk at large detunings.
    """

    num_periods: float = 5.0
    """Number of Rabi periods to sample per trace.

    Sets the total pulse duration: t_max = num_periods / Ω_T.
    Example: 5 periods at 20 MHz → t_max = 250 ns.
    With pts_per_period=5 this gives 25 time points vs. 81 for the Wolff
    defaults — ~3× faster per QUA execution with no loss of BO accuracy.
    Larger → better frequency resolution; smaller → less T2* decay, faster.
    """

    pts_per_period: int = 5
    """Time-domain samples per Rabi period.

    Must satisfy Nyquist (≥ 2). 5 pts/period is sufficient for the BO cost
    function (which only needs Ω_R, A, SNR — not a high-resolution spectrum).
    Total trace length = pts_per_period × num_periods points per QUA execution.
    Wolff et al. use 9; reducing to 5 cuts trace length by ~45% with no loss
    of BO accuracy. Aliasing threshold: (pts_per_period/2) × Ω_T = 50 MHz for
    Ω_T=20 MHz — well above any realistic Rabi frequency near resonance.
    """

    num_shots: int = 100
    """Averages per time-Rabi trace.

    More shots → lower noise → better SNR → better cost estimate.
    Fewer shots → faster iteration → more BO steps in same wall time.
    100 is a reasonable starting point; reduce to 50 if total run time is long.
    """

    min_snr: float = 0.3
    """Minimum SNR for a time-Rabi fit to be accepted (cost < 1e6).

    SNR = fitted_amplitude / residual_RMS. Lower values accept noisier fits,
    which is appropriate for BO (relative ranking matters more than fit quality).
    With 100 shots on a typical transmon, SNR ≈ 0.5–2 near resonance.
    Set higher (e.g. 1.5) only if you want strict fit acceptance.
    """

    pulse_operation: str = "square_drive"
    """QUAM operation name used for the time-Rabi drive pulse.

    IMPORTANT: This operation MUST have a constant (square) amplitude envelope
    so that the Rabi physics is clean. Do NOT use "x180" (Gaussian/DRAG) here.

    TODO: Add a 'square_drive' operation to your QUAM Transmon component:
        qubit.xy.operations["square_drive"] = SquarePulse(amplitude=1.0, length=...)
    Then use amplitude_scale=v_d / 1.0 in the QUA play() call.

    If this operation does not exist, the node will raise a KeyError at runtime.
    """

    # ── Search bounds (relative to QUAM priors) ────────────────────────────────
    freq_search_radius_mhz: float = 200.0
    """±radius in MHz around qubit.xy.RF_frequency (current QUAM prior) for ω_d.

    The total search volume is 2 × freq_search_radius_mhz wide.
    Should encompass the worst-case fabrication spread from design spec.
    Typical transmon frequency scatter: ±100–300 MHz from target.
    """

    min_power_dbm: float = -10.0
    """Lower bound of drive power search, in dBm (absolute, hardware level).

    Should be low enough that the qubit is barely driven (Ω_d << Ω_T).
    For SQMS DR3 fixed-frequency transmons this is approximately -10 dBm;
    the Octave gain for the qubit channel sits ~30 dB higher than the
    resonator channel, so the effective power range is shifted up.
    """

    max_power_dbm: float = 20.0
    """Upper bound of drive power search, in dBm (absolute, hardware level).

    Should not exceed the power where the qubit is strongly over-driven.
    For SQMS DR3 fixed-frequency transmons this is approximately +20 dBm.
    """

    max_amplitude_opx: float = 0.3
    """Maximum OPX output amplitude (V) allowed when setting drive power.

    set_output_power() uses this ceiling to decide whether to increase
    Octave gain (if target power would exceed OPX range) or stay at the
    given amplitude. 0.3 V is a safe default; 0.49 V is the OPX hard limit.
    """

    # ── BO hyperparameters ─────────────────────────────────────────────────────
    n_initial_lhs: int = 12
    """Number of Latin Hypercube Sampling points before BO acquisition starts.

    These quasi-random samples give the GP a global view of the cost landscape
    before exploitation begins. Too few → GP is blind in unexplored regions.
    Too many → wastes hardware time on random sampling.
    12 is good for 2D search; increase to 20 for 4D (joint readout optimization).
    """

    n_bo_iterations: int = 40
    """Maximum BO iterations after the LHS phase.

    Total hardware evaluations ≤ n_initial_lhs + n_bo_iterations = 52.
    At ~2 s/evaluation (100 shots × 20 MHz → 450 ns × 100 + overhead),
    total time ≈ 52 × 2 s ≈ 2 minutes — well under manual spec sweep time.
    """

    acq_function: Literal["EI", "UCB"] = "EI"
    """Acquisition function for selecting the next BO sample point.

    EI (Expected Improvement): exploits known low-cost regions while still
        exploring uncertain ones. Recommended for most cases.
    UCB (Upper Confidence Bound / Lower Confidence Bound in minimization):
        more exploratory, controlled by kappa. Use if EI converges too quickly
        to a local minimum in practice.
    """

    ucb_kappa: float = 2.576
    """Exploration parameter κ for UCB acquisition.

    Higher κ → more exploration. κ=2.576 corresponds to 99% confidence bound.
    Only used when acq_function='UCB'.
    """

    convergence_tolerance_mhz: float = 1.0
    """Early stopping threshold: stop if GP-predicted optimum moves less than
    this (in MHz) between successive BO iterations.

    This is checked on the PREDICTED optimum location (from GP posterior mean),
    not on the observed best point — prevents premature stopping from noise.
    Set to 0.0 to disable early stopping and always run n_bo_iterations.
    """

    gp_noise_level: float = 1e-3
    """Initial noise level for the GP WhiteKernel.

    Tuned by the GP's log-marginal-likelihood optimizer during fitting.
    Represents the aleatoric noise of cost function evaluations.
    Increase if the hardware is very noisy; decrease for cleaner signals.
    """

    # ── Cost function weights ──────────────────────────────────────────────────
    rabi_freq_weight: float = 10.0
    """Weight on |Ω_R − Ω_T| / Ω_T term (resonance penalty).

    This term pushes ω_d toward ω_q AND V_d toward V_π simultaneously,
    because Ω_R = sqrt(Ω_d² + Δ²) depends on both.
    Wolff et al. use weight = 10. Higher → stronger pull toward Ω_T.
    """

    log_amp_weight: float = 3.0
    """Weight on −log(A) term (amplitude reward).

    A = Ω_d / Ω_R peaks at Δ=0 (on resonance). This term differentiates
    the true qubit from spurious modes that might have similar Ω_R but
    much lower oscillation amplitude (e.g., TLS features).
    Wolff et al. use weight = 3.
    """

    log_snr_weight: float = 1.0
    """Weight on −log(SNR) term (readout quality reward).

    Ensures the optimizer steers readout parameters (if optimize_readout_jointly)
    toward good SNR. Lower weight than Rabi/amp terms since readout is
    pre-calibrated by the resonator bringup.
    Wolff et al. use weight = 1.
    """

    log_chi2_weight: float = 2.0
    """Weight on the soft chi² penalty: w·max(0, log(χ²/2)).

    This term makes the cost finite and informative for ALL convergent fits,
    even noisy ones, instead of collapsing them to a 1e6 sentinel.
    This is critical for GP convergence: without it, the GP sees a flat landscape
    everywhere except the handful of clean-oscillation points, causing length
    scales to hit bounds and EI to degenerate to random sampling.

    The penalty is 0 for chi2 ≤ 2 (clean fit), ~1.8 for chi2=5, ~6.4 for chi2=50.
    Set to 0 to disable and revert to the original Wolff formulation.
    """

    # ── Readout settings ───────────────────────────────────────────────────────
    optimize_readout_jointly: bool = False
    """If False (default): use current QUAM readout parameters as-is.
    Readout should already be calibrated by the resonator bringup subgraph.
    The search space is 2D: (ω_d, V_d).

    If True: also optimize ω_r and V_r jointly. Search space becomes 4D.
    Requires more LHS samples (n_initial_lhs ≥ 20) and more BO iterations.
    Only useful if resonator bringup gave poor readout quality.
    """

    readout_freq_radius_mhz: float = 5.0
    """Search radius for ω_r around current QUAM resonator frequency.
    Only used when optimize_readout_jointly=True.
    """

    readout_amp_search_min_db: float = -6.0
    """Lower V_r bound in dB relative to current readout amplitude.
    Only used when optimize_readout_jointly=True.
    """

    readout_amp_search_max_db: float = 6.0
    """Upper V_r bound in dB relative to current readout amplitude.
    Only used when optimize_readout_jointly=True.
    """

    # ── Spurious mode detection (post-convergence sanity check) ───────────────
    run_ramsey_sanity_check: bool = True
    """After BO converges, run a quick Ramsey at the found ω_q to verify
    T2* is physically reasonable (> min_t2star_sanity_ns).
    If T2* is too short, the converged frequency is likely a spurious mode.
    """

    min_t2star_sanity_ns: float = 500.0
    """Minimum T2* (ns) for the post-convergence Ramsey sanity check.
    If measured T2* < this value, outcome="failed" is set and the
    current frequency is blacklisted in QUAM temp_calibration.
    """

    ramsey_sanity_detuning_mhz: float = 1.0
    """Artificial detuning (MHz) for the sanity Ramsey measurement."""

    ramsey_sanity_max_wait_ns: int = 3000
    """Max wait time (ns) for the sanity Ramsey."""

    ramsey_sanity_num_pts: int = 50
    """Number of time points for the sanity Ramsey."""

    ramsey_sanity_num_shots: int = 200
    """Averages for the sanity Ramsey."""
