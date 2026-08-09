"""Analysis utilities for moving-qubit spectroscopy vs flux calibration.

This node is purely explorative: it has no fitting and no state update, so this
module only converts the raw IQ data to physical units and adds convenient
coordinates for plotting.
"""

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert I/Q quadratures to volts and add frequency/flux coordinates.

    The dataset's ``qubit`` dimension carries qubit-pair names (see
    ``execute_qua_program``'s rename of ``qubit_pair`` -> ``qubit``). Both the
    moving and stationary qubit readouts are converted to volts using each
    qubit's own readout pulse length, since the two resonators are generally
    different.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    qubit_roles_map = node.namespace["qubit_roles_map"]
    pair_names = [qp.name for qp in qubit_pairs]

    moving_readout_length = xr.DataArray(
        [qubit_roles_map[qp.name].moving.resonator.operations["readout"].length for qp in qubit_pairs],
        coords=[("qubit", pair_names)],
    )
    stationary_readout_length = xr.DataArray(
        [qubit_roles_map[qp.name].stationary.resonator.operations["readout"].length for qp in qubit_pairs],
        coords=[("qubit", pair_names)],
    )
    ds = ds.assign(
        {
            "I_moving": ds.I_moving * 2**12 / moving_readout_length,
            "Q_moving": ds.Q_moving * 2**12 / moving_readout_length,
            "I_stationary": ds.I_stationary * 2**12 / stationary_readout_length,
            "Q_stationary": ds.Q_stationary * 2**12 / stationary_readout_length,
        }
    )

    # Amplitude of the readout signal for each qubit (no phase needed for this explorative node).
    ds["IQ_abs_moving"] = np.abs(ds.I_moving + 1j * ds.Q_moving)
    ds["IQ_abs_stationary"] = np.abs(ds.I_stationary + 1j * ds.Q_stationary)

    # The moving qubit's drive frequency is swept, but centered on the stationary qubit's
    # frequency (not the moving qubit's own bare frequency) — see create_qua_program.
    full_freq = np.array([ds.detuning + qubit_roles_map[qp.name].stationary.f_01 for qp in qubit_pairs])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}

    # The flux sweep is on the moving qubit's flux line, centered on its own independent idle
    # offset — each pair's moving qubit can have a different idle offset, so the absolute flux
    # bias is a per-qubit coordinate built from the shared relative sweep ("flux_bias").
    full_flux = np.array(
        [ds.flux_bias + qubit_roles_map[qp.name].moving.z.independent_offset for qp in qubit_pairs]
    )
    ds = ds.assign_coords(full_flux=(["qubit", "flux_bias"], full_flux))
    ds.full_flux.attrs = {"long_name": "Moving qubit flux bias", "units": "V"}

    current = ds.full_flux / node.parameters.input_line_impedance_in_ohm
    ds = ds.assign_coords({"current": (["qubit", "flux_bias"], current.data)})
    ds.current.attrs["long_name"] = "Current"
    ds.current.attrs["units"] = "A"
    attenuation_factor = 10 ** (-node.parameters.line_attenuation_in_db / 20)
    attenuated_current = ds.current * attenuation_factor
    ds = ds.assign_coords({"attenuated_current": (["qubit", "flux_bias"], attenuated_current.values)})
    ds.attenuated_current.attrs["long_name"] = "Attenuated Current"
    ds.attenuated_current.attrs["units"] = "A"

    return ds
