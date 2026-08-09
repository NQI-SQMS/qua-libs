"""Analysis utilities for conditional qubit spectroscopy vs flux calibration.

This node is purely explorative: it has no fitting and no state update, so this
module only converts the raw IQ data to physical units and adds convenient
coordinates for plotting.
"""

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q quadratures to volts and add frequency/current coordinates.

    The dataset's ``qubit`` dimension carries qubit-pair names (see
    ``execute_qua_program``'s rename of ``qubit_pair`` -> ``qubit``). Both the
    control and target readouts are converted to volts using each qubit's own
    readout pulse length, since the two resonators are generally different.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    pair_names = [qp.name for qp in qubit_pairs]

    control_readout_length = xr.DataArray(
        [qp.qubit_control.resonator.operations["readout"].length for qp in qubit_pairs],
        coords=[("qubit", pair_names)],
    )
    target_readout_length = xr.DataArray(
        [qp.qubit_target.resonator.operations["readout"].length for qp in qubit_pairs],
        coords=[("qubit", pair_names)],
    )
    ds = ds.assign(
        {
            "I_control": ds.I_control * 2**12 / control_readout_length,
            "Q_control": ds.Q_control * 2**12 / control_readout_length,
            "I_target": ds.I_target * 2**12 / target_readout_length,
            "Q_target": ds.Q_target * 2**12 / target_readout_length,
        }
    )

    # Amplitude of the readout signal for each qubit (no phase needed for this explorative node).
    ds["IQ_abs_control"] = np.abs(ds.I_control + 1j * ds.Q_control)
    ds["IQ_abs_target"] = np.abs(ds.I_target + 1j * ds.Q_target)

    # Only the target qubit's frequency is swept, so the RF frequency axis is built from it.
    full_freq = np.array([ds.detuning + qp.qubit_target.xy.RF_frequency for qp in qubit_pairs])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    # The flux sweep is on the target qubit's flux line.
    current = ds.flux_bias / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"

    return ds
