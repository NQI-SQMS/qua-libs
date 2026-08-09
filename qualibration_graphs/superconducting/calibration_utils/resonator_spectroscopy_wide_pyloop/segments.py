"""Segment planner for wide-band resonator spectroscopy.

Splits an absolute RF scan range [f_start_hz, f_stop_hz] into a list of
LO segments, each with a single MW-FEM upconverter/downconverter frequency
and a one-sided IF sweep. Avoids the |IF| <= 5 MHz hardware dead zone
(default margin 20 MHz) and respects the MW-FEM band LO limits:

    band 1: 50 MHz  - 5.5 GHz
    band 2: 4.5 GHz - 7.5 GHz
    band 3: 6.5 GHz - 10.5 GHz

LO is constrained strictly inside the selected band's LO range; the RF
reach per band is extended ±(if_max - if_dead_zone) past the band edges
because the IF window can push the carrier outside the LO band. The
default `if_max` is 400 MHz (SNR-guaranteed window); the OPX1000 hardware
ceiling is 500 MHz, so callers can opt into the noisier 400-500 MHz IFs.

For each segment the planner picks a band from `bands_priority` (first
that works), clamps LO to the band edge when needed, and recomputes the
IF axis from the actual (lo, seg_f_start, seg_f_end). Different segments
in the same scan can therefore use different bands — cross-band scans
like [5.5, 8.8] GHz work in a single call.
"""

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np


BAND_LIMITS_HZ = {
    1: (50e6, 5.5e9),
    2: (4.5e9, 7.5e9),
    3: (6.5e9, 10.5e9),
}


@dataclass
class Segment:
    """One LO segment of a wide spectroscopy scan.

    `dfs_hz` are the (signed) IF offsets sent to `update_frequency`.
    `rf_hz = lo_hz + dfs_hz` gives the absolute RF axis covered.
    Points sorted by ascending RF. `band` is the MW-FEM band index the
    LO must be configured for during this segment.
    """

    lo_hz: int
    dfs_hz: np.ndarray   # int Hz, length n_points
    rf_hz: np.ndarray    # float Hz, sorted ascending
    band: int

    @property
    def n_points(self) -> int:
        return len(self.dfs_hz)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _reachable_rf_intervals(
    bands: Iterable[int], if_dead_zone_hz: float, if_max_hz: float
) -> list[tuple[float, float]]:
    """RF intervals reachable per band (LO inside band, |IF| in [dz, if_max])."""
    spans = []
    for b in bands:
        lo_lo, lo_hi = BAND_LIMITS_HZ[b]
        spans.append((lo_lo - (if_max_hz - if_dead_zone_hz), lo_hi + (if_max_hz - if_dead_zone_hz)))
    return spans


def _missing_subranges(
    f_start: float, f_stop: float, intervals: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return the sub-intervals of [f_start, f_stop] not covered by `intervals`."""
    # Merge intervals
    merged = sorted(intervals)
    out = []
    cur = f_start
    for lo, hi in merged:
        if hi < cur:
            continue
        if lo > cur:
            out.append((cur, min(lo, f_stop)))
        cur = max(cur, hi)
        if cur >= f_stop:
            break
    if cur < f_stop:
        out.append((cur, f_stop))
    return [(a, b) for a, b in out if b > a]


def can_band_cover(
    band: int,
    f_start_hz: float,
    f_stop_hz: float,
    if_dead_zone_hz: float = 20e6,
    if_max_hz: float = 400e6,
) -> bool:
    """True iff a single MW-FEM `band` can physically cover [f_start, f_stop].

    Coverage means: for every RF in the range, there exists an LO inside the
    band's nominal LO range such that |IF| stays inside [if_dead_zone, if_max].
    """
    if band not in BAND_LIMITS_HZ:
        raise ValueError(f"band must be 1, 2, or 3; got {band}")
    intervals = _reachable_rf_intervals([band], if_dead_zone_hz, if_max_hz)
    return not _missing_subranges(f_start_hz, f_stop_hz, intervals)


def _try_band_for_segment(
    band: int,
    seg_f_start: float,
    seg_f_end: float,
    if_dead_zone_hz: float,
    if_max_hz: float,
) -> Union[tuple[int, float, float], None]:
    """Try to fit one segment in one band. Return (lo_hz, if_lo, if_hi) on success.

    `if_lo` and `if_hi` are signed (both positive for LO-below, both
    negative-ascending for LO-above). On failure returns None.
    """
    lo_lo, lo_hi = BAND_LIMITS_HZ[band]

    # Try LO below the segment (positive IFs).
    lo_below = _clamp(seg_f_start - if_dead_zone_hz, lo_lo, lo_hi)
    if_lo = seg_f_start - lo_below
    if_hi = seg_f_end - lo_below
    if if_dead_zone_hz <= if_lo and if_hi <= if_max_hz:
        return (int(round(lo_below)), if_lo, if_hi)

    # Try LO above the segment (negative IFs, kept ascending).
    lo_above = _clamp(seg_f_end + if_dead_zone_hz, lo_lo, lo_hi)
    if_hi_neg = seg_f_start - lo_above       # most negative
    if_lo_neg = seg_f_end - lo_above         # least negative (closer to 0)
    # if_hi_neg < if_lo_neg <= -dead_zone < 0, |if_hi_neg| <= if_max
    if (-if_max_hz) <= if_hi_neg and if_lo_neg <= -if_dead_zone_hz:
        return (int(round(lo_above)), if_hi_neg, if_lo_neg)

    return None


def plan_segments(
    f_start_hz: float,
    f_stop_hz: float,
    step_hz: float,
    bands_priority: Union[int, Iterable[int]],
    if_dead_zone_hz: float = 20e6,
    if_max_hz: float = 400e6,
) -> list[Segment]:
    """Plan one-sided IF segments covering exactly [f_start_hz, f_stop_hz].

    Args:
        f_start_hz, f_stop_hz: requested absolute RF scan range (Hz).
        step_hz: IF step size (Hz). Determines points per segment.
        bands_priority: ordered list of MW-FEM bands (1, 2, or 3) to try
            per segment. The first band whose LO range + IF window can
            cover the segment is used. A bare int is accepted for back-
            compatibility (wrapped to `[int]`).
        if_dead_zone_hz: minimum |IF| (Hz). Default 20 MHz; hardware floor 5 MHz.
        if_max_hz: maximum |IF| (Hz). Default 400 MHz, where readout SNR is
            well-characterised. Hardware ceiling is 500 MHz; callers can pass
            up to 500e6 if they accept degraded signal quality above 400 MHz.

    Returns:
        List of Segment, ordered by ascending RF, whose union covers exactly
        [f_start_hz, f_stop_hz] (no overshoot). Per-segment point counts may
        vary by ±1 when LO is clamped to a band edge.
    """
    if f_stop_hz <= f_start_hz:
        raise ValueError(f"f_stop_hz ({f_stop_hz}) must be > f_start_hz ({f_start_hz})")

    if isinstance(bands_priority, int):
        bands_priority = [bands_priority]
    bands_priority = list(bands_priority)
    for b in bands_priority:
        if b not in BAND_LIMITS_HZ:
            raise ValueError(f"band must be 1, 2, or 3; got {b}")
    if not bands_priority:
        raise ValueError("bands_priority must contain at least one band")

    max_seg_width_hz = if_max_hz - if_dead_zone_hz
    if max_seg_width_hz <= 0:
        raise ValueError("if_max_hz must be greater than if_dead_zone_hz")

    # qualang_tools.from_array detects linear-vs-log spacing by comparing
    # std(diff(array)) (linear) and std(array[1:]/array[:-1]) (log). With
    # integer IFs from np.linspace over a non-step-multiple span, the int
    # rounding makes std(diffs) > 0 by ~0.5 Hz; from_array then mistakenly
    # picks log, multiplies var by ~1+eps per iteration, and the real OPX
    # IFs no longer match seg.rf_hz. To avoid this we snap segment widths
    # AND each segment's IF axis to integer multiples of step_hz, then build
    # dfs_hz with np.arange so adjacent diffs are exactly step_int.
    step_int = int(round(step_hz))
    if step_int <= 0:
        raise ValueError(f"step_hz must be >= 1 Hz; got {step_hz}")

    # Snap f_start to the step grid (preserves user input to within < step).
    f_start_hz = int(round(f_start_hz / step_int) * step_int)
    f_stop_hz = int(round(f_stop_hz / step_int) * step_int)

    # Up-front reach check across the union of allowed bands.
    intervals = _reachable_rf_intervals(bands_priority, if_dead_zone_hz, if_max_hz)
    missing = _missing_subranges(f_start_hz, f_stop_hz, intervals)
    if missing:
        miss_str = ", ".join(f"[{a/1e9:.3f}, {b/1e9:.3f}] GHz" for a, b in missing)
        raise ValueError(
            f"Requested range [{f_start_hz/1e9:.3f}, {f_stop_hz/1e9:.3f}] GHz "
            f"includes unreachable sub-range(s) {miss_str} given bands "
            f"{bands_priority} and IF window ±{if_max_hz/1e6:.0f} MHz "
            f"(dead zone {if_dead_zone_hz/1e6:.0f} MHz)."
        )

    total_range_hz = f_stop_hz - f_start_hz
    n_segs = max(1, int(np.ceil(total_range_hz / max_seg_width_hz)))
    # Snap seg width to a multiple of step_int. Each segment then has an
    # exact integer step and the same point count.
    seg_width_pts = max(2, int(round(total_range_hz / n_segs / step_int)))
    seg_width_hz = seg_width_pts * step_int
    # Actual f_stop after snapping may overshoot the request by up to
    # (n_segs * step_int) (< step per segment); negligible at 0.1 MHz step.

    segments: list[Segment] = []
    for k in range(n_segs):
        seg_f_start = f_start_hz + k * seg_width_hz
        seg_f_end = seg_f_start + seg_width_hz

        # Score every viable band by the LO's distance to the nearest band
        # edge — higher margin is safer (better filter response, no edge
        # clamping needed). Tiebreaker: earlier position in bands_priority,
        # so segments that fit comfortably in the current port band stay
        # there and don't trigger unnecessary band swaps.
        candidates = []  # (margin, prio_idx, lo_hz, if_lo, if_hi, band)
        for prio_idx, band in enumerate(bands_priority):
            result = _try_band_for_segment(
                band, seg_f_start, seg_f_end, if_dead_zone_hz, if_max_hz
            )
            if result is None:
                continue
            lo_hz_cand, if_lo_cand, if_hi_cand = result
            lo_lo_b, lo_hi_b = BAND_LIMITS_HZ[band]
            margin = min(lo_hz_cand - lo_lo_b, lo_hi_b - lo_hz_cand)
            candidates.append(
                (margin, prio_idx, lo_hz_cand, if_lo_cand, if_hi_cand, band)
            )

        if not candidates:
            raise ValueError(
                f"Segment [{seg_f_start/1e9:.3f}, {seg_f_end/1e9:.3f}] GHz "
                f"cannot fit any of bands {bands_priority} within "
                f"±{if_max_hz/1e6:.0f} MHz IF (dead zone {if_dead_zone_hz/1e6:.0f} MHz)."
            )

        candidates.sort(key=lambda c: (-c[0], c[1]))
        _, _, lo_hz, if_lo, if_hi, band = candidates[0]
        # Snap if_lo onto the step grid so dfs_hz has exact integer step.
        # seg_f_start and lo_hz are both step-aligned (f_start snapped above,
        # band-edge clamps are coarse), so this rounding shifts by < step.
        if_lo_int = int(round(if_lo / step_int)) * step_int
        dfs_hz = (if_lo_int + np.arange(seg_width_pts) * step_int).astype(int)
        rf_hz = (lo_hz + dfs_hz).astype(float)
        segments.append(Segment(lo_hz=lo_hz, dfs_hz=dfs_hz, rf_hz=rf_hz, band=band))

    return segments


def concatenate_rf_axis(segments: list[Segment]) -> np.ndarray:
    """Return the concatenated, ascending RF axis covered by all segments (Hz)."""
    return np.concatenate([seg.rf_hz for seg in segments])


def crop_to_requested_range(
    rf_hz: np.ndarray,
    values: np.ndarray,
    f_start_hz: float,
    f_stop_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim a concatenated RF/value pair to [f_start_hz, f_stop_hz].

    `values` may be 1D matching rf_hz, or 2D with the last axis matching rf_hz.
    """
    mask = (rf_hz >= f_start_hz) & (rf_hz <= f_stop_hz)
    return rf_hz[mask], values[..., mask]
