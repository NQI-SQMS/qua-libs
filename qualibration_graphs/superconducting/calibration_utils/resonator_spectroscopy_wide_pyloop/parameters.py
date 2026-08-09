"""Parameter definitions for wide-band resonator spectroscopy calibration."""

from typing import List, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Wide resonator spectroscopy parameters.

    The wide scan covers an absolute RF range [freq_start_ghz, freq_stop_ghz]
    using one MW-FEM LO segment after another. Each segment scans up to
    (if_max_mhz - if_dead_zone_mhz) MHz of one-sided IF; pieces of the range
    falling under |IF| <= if_dead_zone_mhz are unreachable in a single LO setting
    and are filled by the neighbouring segment(s).
    """

    num_shots: int = 100
    """Number of averages per IF point. Default 100."""

    global_readout_amp: Optional[float] = None
    """Single uniform readout amplitude for the wide sweep (the literal pulse amplitude value,
    same units as the readout operation's stored ``amplitude``).

    Each readout feedline is swept ONCE by one probe qubit, and every other qubit on that line
    reads its own resonator off that single trace. Leave BLANK (None) to probe with the
    representative qubit's own stored readout amplitude. Set a value to drive the whole line at
    ONE uniform amplitude instead — the probe's readout pulse amplitude is overridden directly
    to this value for the sweep (and reverted afterwards). It is the amplitude itself, NOT a
    scale factor; bounded only by the hardware DAC range (config validation will reject a value
    that is too large). Pick it large enough that every resonator on the line shows a clear dip
    (a probe tuned for one qubit can leave another qubit's resonator too shallow to detect)."""

    # --- Wide-scan range ---
    freq_start_ghz: float = 4.5
    """Lower edge of the absolute RF scan range (GHz). Default 4.5."""
    freq_stop_ghz: float = 6.0
    """Upper edge of the absolute RF scan range (GHz). Default 6.0."""
    frequency_step_in_mhz: float = 0.1
    """IF step size (MHz). Determines points per segment. Default 0.1."""

    # --- Segmentation knobs ---
    if_dead_zone_mhz: float = 20.0
    """Minimum |IF| magnitude (MHz). Hardware floor is 5 MHz; 20 leaves headroom."""
    if_max_mhz: float = 400.0
    """Maximum |IF| magnitude (MHz). Default 400 MHz, where readout signal
    quality is well-characterised. The OPX1000 hardware allows up to 500 MHz,
    but the 400-500 MHz region has degraded SNR and is not guaranteed."""

    band: Optional[int] = None
    """If set (1, 2, or 3), force every segment to use this MW-FEM band when
    physically possible (LO inside the band, IFs within ±if_max_mhz). If the
    requested scan range cannot be fully covered by this band alone, the
    planner falls back to automatic per-segment band selection (max LO-edge
    margin across bands 1/2/3). Default None = automatic from the start."""

    # --- Multi-dip detection / assignment ---
    min_dip_prominence_db: float = 2.0
    """Minimum prominence (dB) for a dip to qualify as a candidate."""
    proximity_tolerance_mhz: float = 50.0
    """Max distance (MHz) between a qubit's initial RF_frequency and an assigned dip.
    Qubits with no candidate within this radius are marked failed."""

    # --- Re-fit overrides (used together with load_data_id) ---
    re_fit_resonators: Optional[List[str]] = None
    """Qubit names to re-fit with a manually specified window, e.g. ["qA1", "qD3"].
    Must be the same length as re_fit_centers_ghz and re_fit_span_mhz."""
    re_fit_centers_ghz: Optional[List[float]] = None
    """Absolute RF center frequency (GHz) for each qubit in re_fit_resonators."""
    re_fit_span_mhz: Optional[List[float]] = None
    """Fit span (MHz) around the center for each qubit in re_fit_resonators."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameter class for wide resonator spectroscopy calibration node."""
