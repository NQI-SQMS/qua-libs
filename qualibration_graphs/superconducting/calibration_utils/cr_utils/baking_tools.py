import numpy as np

from typing import Dict, List
from qualang_tools.bakery import baking
from qualang_tools.bakery.bakery import Baking
from qualang_tools.config.waveform_tools import (
    drag_gaussian_pulse_waveforms,
    drag_cosine_pulse_waveforms,
    flattop_gaussian_waveform,
)


def generate_kwargs_list(**kwargs):
    """
    Expand kwargs where values may be scalars or non-empty list/ndarray.
    - Empty lists/arrays are invalid.
    - If any list/array is present, all such sequences must share the same length.
    Returns a list of per-item kwargs dicts.
    """
    # allow numpy arrays if numpy is available, else only lists
    seq_types = (list, np.ndarray)

    seq = {k: v for k, v in kwargs.items() if isinstance(v, seq_types)}
    scal = {k: v for k, v in kwargs.items() if k not in seq}

    # validate sequences (non-empty, equal lengths)
    for k, v in seq.items():
        if len(v) == 0:
            raise ValueError(f"Argument '{k}' cannot be an empty list/array")
    if seq:
        lens = {len(v) for v in seq.values()}
        if len(lens) != 1:
            raise ValueError(f"Inconsistent sequence lengths: {lens}")

    if not seq:  # only scalars
        return [dict(scal)]

    keys = tuple(seq.keys())
    cols = [seq[k] for k in keys]
    return [{**dict(zip(keys, row)), **scal} for row in zip(*cols)]


def flatten_dict_str(d: dict) -> str:
    return "/".join(f"{k}{v}" for k, v in d.items())


def const_waveforms(amplitude, length):
    return [amplitude] * length


def bake_waveforms(
    wf_type, qubit_pairs, config, **kwargs
) -> Dict[str, List[Baking]]:
    pulse_segments = {}  # Stores the baking objects

    # create kwargs for each
    kwargs_list = generate_kwargs_list(**kwargs)

    # generate based on the waveform type
    if wf_type == "square":
        wf_func = const_waveforms
    elif wf_type == "cosine":
        wf_func = drag_gaussian_pulse_waveforms
    elif wf_type == "gauss":
        wf_func = drag_cosine_pulse_waveforms
    elif wf_type == "flattop":
        wf_func = flattop_gaussian_waveform
    else:
        raise NotImplementedError(f"{wf_type} waveform is not supported")

    # bake the zz waveform for all the qubit pairs
    for multiplexed_qubit_pairs in qubit_pairs.batch():
        for i, qp in multiplexed_qubit_pairs.items():
            qc = qp.qubit_control
            qt = qp.qubit_target
            cr = qp.cr_drive
            pulse_segments[qp.name] = []

            for _kwargs in kwargs_list:
                kwargs_str = flatten_dict_str(_kwargs)
                with baking(config, padding_method="right") as b:
                    wf = wf_func(**_kwargs)
                    if len(wf) == 2 and isinstance(wf[0], list):
                        wf_I, wf_Q = wf
                    else:
                        wf_I = wf
                        wf_Q = const_waveforms(0, len(wf))
                    assert len(wf_I) == len(wf_Q)
                    b.add_op(f"{wf}_{kwargs_str}", cr.name, [wf_I, wf_Q])
                    b.add_op(
                        f"cr_{wf}_{qp.name}_{kwargs_str}",
                        qt.xy_detuned.name,
                        [wf_I, wf_Q],
                    )

                    b.play(f"{wf}_{kwargs_str}", cr.name)
                    b.play(f"cr_{wf}_{qp.name}_{kwargs_str}", qt.xy.name)

                # Append the baking object in the list to call it from the QUA program
                pulse_segments[qp.name].append(b)

    return pulse_segments
