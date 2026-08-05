"""Parameters for qubit spectroscopy versus drive power (amplitude) calibration.

Only the experiment knobs and a few meaningful operating-point / feature toggles are
exposed. The fit hyperparameters (peak-finder prominence, power-broadening factor,
search windows, anharmonicity sanity bounds, fringe sensitivity, ...) are handled
automatically inside ``analysis.py`` with robust defaults, so non-expert users do not
face a wall of fit knobs. ``auto_tune`` (default True) self-selects the per-qubit
peak-finder prominence. Power-user overrides for the hidden values live as module
constants at the top of ``analysis.py``.
"""

from typing import Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for qubit spectroscopy versus drive power."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_span_in_mhz: float = 100
    """Span of frequencies to sweep in MHz. Default is 100 MHz."""
    frequency_step_in_mhz: float = 0.25
    """Step size for frequency sweep in MHz. Default is 0.25 MHz."""

    # --- Drive pulse ---
    operation: str = "saturation"
    """Qubit xy operation used to drive the spectroscopy. Default is "saturation"."""
    operation_len_in_ns: Optional[int] = None
    """Length of the operation in nanoseconds. Default is the predefined pulse length. If set, this length is
    also persisted to the operation on the state update (the fitted power/amplitude is only valid for the length
    actually swept, so the length must travel with it)."""

    # --- Adaptive saturation-duration controls ---
    duration_mode: str = "fixed"
    """How the saturation-pulse duration is set across the amplitude sweep:
    "fixed" (one duration for all amplitudes — the classic behaviour) or "constant_angle"
    (duration scaled inversely with amplitude so Omega*t is constant, which removes the
    coherent Rabi-nutation fringing the fixed scheme can show at high power). Default "fixed"."""
    base_len_ns: int = 100
    """constant_angle mode: the saturation duration (ns) at the HIGHEST amplitude (a = a_max).
    The duration at amplitude a is base_len_ns * a_max / a. Choose directly (no pi/T1/T2
    needed); ~100-200 ns is a sensible start. Default 100. (Ignored in fixed mode.)"""
    max_len_ns: int = 4000
    """constant_angle mode: hard upper cap (ns) on the (inversely-scaled) duration at the
    lowest amplitudes, to bound acquisition time. Default 4000 ns. (Ignored in fixed mode.)"""

    # --- Drive-power sweep (mirrors resonator_spectroscopy_vs_amplitude) ---
    max_power_dbm: int = -20
    """Maximum qubit-drive power level in dBm (top of the sweep). Default is -20 dBm."""
    min_power_dbm: int = -55
    """Minimum qubit-drive power level in dBm (bottom of the sweep). Default is -55 dBm."""
    num_power_points: int = 100
    """Number of points of the drive-power axis. Default is 100."""
    max_amp: float = 0.1
    """Maximum drive waveform amplitude (V) for the saturation operation. Default is 0.1."""

    # --- Operating point ---
    power_below_saturation_db: float = 1.0
    """The optimal drive power is set this many dB BELOW the saturation/power-broadening onset: just inside the
    onset so the GE line still has good SNR/width, while staying out of strong power broadening / the 2-photon
    g->f regime. Default is 1 dB."""

    # --- Auto-tuning ---
    auto_tune: bool = True
    """If True, the analysis auto-selects the peak-finder prominence per qubit, so the user does not have to
    hand-tune it. It sweeps candidate prominences high->low and keeps the most selective one that still detects
    a consistent peak. The remaining fit hyperparameters have robust defaults inside analysis.py. Default True."""

    # --- Feature toggles ---
    fit_ef_transition: bool = True
    """If True, also fit the 2-photon g->f/2 transition that appears at higher drive power near
    detuning = -anharmonicity/2, and report the measured anharmonicity |alpha| = 2*(f_GE - f_2photon).
    Uses the stored ``qubit.anharmonicity`` to know where to look. Default is True."""
    update_anharmonicity: bool = True
    """If True, propose a state update of ``qubit.anharmonicity`` from the EF fit (GUI-gated via
    record_state_updates; only applied when the EF fit passes the internal sanity gate). Default is True."""
    ge_low_power_first: bool = True
    """If True, read the GE frequency at the lowest powers where the in-window peak is detected (least power
    broadening / AC-Stark), rather than at the optimal power. Default is True."""
    update_pulses_amplitude: bool = False
    """If True, also update the saturation pulse amplitude from the fitted linewidth (using the internal
    target peak width in analysis.py). Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for qubit spectroscopy versus drive power calibration."""
