"""Shared helper for calibration nodes that set an output power without touching gain.

The generic `set_output_power` API in quam-builder will, by default, solve for a new
Octave gain / full_scale_power_dbm to hit the requested power. Calibration nodes that
share a physical channel (Octave RF output / OPX1000 FEM port) across multiple
qubits or resonators must never change that gain at runtime -- it is set once when the
QUAM state is populated. This helper locks the channel's current gain/full_scale_power
and only ever solves for the operation's amplitude.

Note: the generic `set_output_power_iq_channel` only validates the requested
`max_amplitude` input, not the `amplitude` it computes when `gain` is given explicitly
(the "locked" path used here). Likewise, `set_output_power_mw_channel` will silently
bump `full_scale_power_dbm` if the computed amplitude would exceed `max_amplitude`, even
when `full_scale_power_dbm` was passed explicitly. So we pre-validate the amplitude
ourselves and raise a clear error before calling into the generic function, rather than
relying on it to reject (or worse, silently accept) an out-of-range waveform.
"""

from typing import Optional

from qualang_tools.units import unit

from quam_builder.tools.power_tools import calculate_voltage_scaling_factor

_OPX_MAX_AMPLITUDE = 0.5


def set_locked_output_power(channel, power_in_dbm: float, operation: Optional[str] = "readout"):
    """Set `power_in_dbm` on `channel` without changing its Octave gain / full_scale_power_dbm.

    Parameters:
        channel: An XY drive or readout resonator channel (IQ + Octave, or MW-FEM).
        power_in_dbm (float): Desired output power in dBm.
        operation (Optional[str]): Name of the operation to configure, default is "readout".

    Raises:
        ValueError: If `power_in_dbm` cannot be reached at the channel's current
            gain/full_scale_power_dbm without exceeding the hardware amplitude range.
    """
    if hasattr(channel, "frequency_converter_up"):
        gain = channel.frequency_converter_up.gain
        u = unit(coerce_to_integer=True)
        amplitude = u.dBm2volts(power_in_dbm - gain)
        if not -_OPX_MAX_AMPLITUDE <= amplitude < _OPX_MAX_AMPLITUDE:
            raise ValueError(
                f"Reaching {power_in_dbm} dBm on {channel.name} at its current Octave gain "
                f"of {gain} dB requires an amplitude of {amplitude:.3f} V, which is outside "
                f"the OPX+ range [-{_OPX_MAX_AMPLITUDE}, {_OPX_MAX_AMPLITUDE}). Lower the "
                f"requested power, or re-run the QUAM population step with a different gain "
                f"for this channel."
            )
        return channel.set_output_power(power_in_dbm=power_in_dbm, gain=gain, operation=operation)

    full_scale_power_dbm = channel.opx_output.full_scale_power_dbm
    amplitude = calculate_voltage_scaling_factor(
        fixed_power_dBm=full_scale_power_dbm, target_power_dBm=power_in_dbm
    )
    if power_in_dbm > full_scale_power_dbm or amplitude > 1:
        raise ValueError(
            f"Reaching {power_in_dbm} dBm on {channel.name} at its current "
            f"full_scale_power_dbm of {full_scale_power_dbm} dBm requires an amplitude of "
            f"{amplitude:.3f}, which exceeds the channel's range. Lower the requested power, "
            f"or re-run the QUAM population step with a different full_scale_power_dbm for "
            f"this channel."
        )
    return channel.set_output_power(
        power_in_dbm=power_in_dbm,
        full_scale_power_dbm=full_scale_power_dbm,
        max_amplitude=1,
        operation=operation,
    )
