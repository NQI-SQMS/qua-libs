"""Wide-band Resonator Spectroscopy — one job per LO segment, Python-loop scheme.

Sweeps an absolute RF range [freq_start_ghz, freq_stop_ghz] by running one
short QUA program per MW-FEM LO segment. For each segment, Python uses
`tracked_updates` to swap in the segment's LO on every qubit's readout
output port (the input-port downconverter follows by reference in QUAM),
regenerates the config, opens a QM, executes the per-segment program, and
fetches its dataset. Per-segment datasets are concatenated along RF.

See the companion node `1Q_03_resonator_spectroscopy_wide_pause_resume.py`
for the single-job scheme that retunes via `job.set_converter_frequency`
between QUA `pause()`s.

Prerequisites:
    - Having calibrated the IQ mixer/MW-FEM connected to the readout line.
    - Having calibrated the time of flight, offsets, and gains.
    - Having initialised the QUAM state parameters for the readout pulse and
      depletion time.
    - Having specified the desired flux point if relevant (qubit.z.flux_point).
    - All listed qubits' readout ports must share the same MW-FEM band
      (1/2/3), since the LO is shared across qubits per segment.

State update:
    - q.resonator.f_01 & q.resonator.RF_frequency (per assigned dip)
"""

# %% {Imports}
from contextlib import ExitStack
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from quam.utils import string_reference

from qualibration_libs.core import tracked_updates

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.resonator_spectroscopy_wide_pyloop import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_amplitude_with_fit,
    plot_local_amplitude_with_fit,
    plot_raw_phase,
    plot_detrended_phase,
    plot_iq_circle,
    plan_segments,
    can_band_cover,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.parameters.experiment import BatchableList
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# %% {Node initialisation}
description = """
        WIDE RESONATOR SPECTROSCOPY (python loop)
Sweeps an absolute RF range using one MW-FEM LO segment per QUA job.
Python's outer loop tracked_updates each qubit's readout LO per segment,
regenerates the config, executes one program per segment, and concatenates
the per-segment datasets. Identifies multiple dip candidates and assigns
each to the listed qubits by proximity to their current resonator.RF_frequency.
"""

node = QualibrationNode[Parameters, Quam](
    name="02_resonator_spectroscopy_wide_pyloop",
    description=description,
    parameters=Parameters(),
    machine = Quam.load()
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging."""
    pass


# Stash re-fit overrides before load_from_id() can overwrite node.parameters
_stored_re_fit_resonators = node.parameters.re_fit_resonators
_stored_re_fit_centers_ghz = node.parameters.re_fit_centers_ghz
_stored_re_fit_span_mhz = node.parameters.re_fit_span_mhz

# node.machine = Quam.load()


# %% {Helpers}
def _qubit_band(q) -> int:
    """Return the MW-FEM band integer for a qubit's readout output port."""
    return int(q.resonator.opx_output.band)


def _group_qubits_by_feedline(qubits):
    """Group qubits by readout feedline = the reference of ``resonator.opx_output``.

    All resonators on one physical feedline appear in a single wide |I+jQ| trace of
    that line, so the line only needs to be swept ONCE. Returns a dict
    ``{feedline_reference -> [qubits on it]}`` in first-seen order (qubit list order).
    """
    groups = {}
    for q in qubits:
        key = q.resonator.opx_output.get_reference()
        groups.setdefault(key, []).append(q)
    return groups


def _feedline_representatives(node, qubits):
    """One probe qubit per readout feedline, as a ``BatchableList`` honoring ``multiplexed``.

    The QUA program measures only these representatives (one uniform sweep per feedline);
    every other listed qubit is fanned out from its feedline-representative's trace after
    the fetch (see ``execute_qua_program``). Cuts the measurement from
    ``N_qubits x npts x navg`` to ``N_feedlines x npts x navg`` with no change to the
    analysis/plotting/state-update path, which stays keyed by qubit name over the full list.

    The probe per feedline is the qubit with the LARGEST stored readout amplitude. A strong
    probe drives every resonator on the shared line hard enough to show a clear dip (a weak
    probe leaves higher-power resonators too shallow to detect), and it maximises the
    headroom for ``global_readout_amp`` — the uniform target is reachable via the QUA
    amplitude prefactor (-2, 2) only when the probe's base amplitude is at least target/2.
    """
    groups = _group_qubits_by_feedline(qubits)

    def _readout_amp(q):
        return float(q.resonator.operations["readout"].amplitude)

    reps = [max(qs, key=_readout_amp) for qs in groups.values()]
    # Map every listed qubit to the NAME of its feedline's representative (the probe whose
    # trace it is fanned out from). Built from the actual reps (not qs[0]) so it stays correct
    # whatever rule picks the probe.
    rep_name_for_qubit = {
        q.name: rep.name for qs, rep in zip(groups.values(), reps) for q in qs
    }
    multiplexed = getattr(node.parameters, "multiplexed", False)
    batch_groups = [list(range(len(reps)))] if multiplexed else [[i] for i in range(len(reps))]
    return BatchableList(reps, batch_groups), groups, rep_name_for_qubit


def _tracked_set_unless_reference(quam_obj, tracked_obj, attr_name, value):
    """Tracked-update an attribute only if it isn't a QUAM reference.

    If the field is a reference (e.g. downconverter_frequency points at
    upconverter_frequency, or vice versa), writing to it would raise per
    QUAM's _is_valid_setattr, AND would replace the reference with a raw
    value on revert. Skipping the write is safe because the reference
    follows the source field that is being set elsewhere.
    """
    raw = quam_obj.get_raw_value(attr_name)
    if string_reference.is_reference(raw):
        return
    setattr(tracked_obj, attr_name, value)


def _build_segments(node: QualibrationNode[Parameters, Quam], qubits) -> list:
    """Plan LO segments shared across all listed qubits.

    Band selection: if `node.parameters.band` is set and that band can
    physically cover the full requested range (LO inside band, IFs within
    ±if_max_mhz), every segment uses it. Otherwise — either band unset, or
    user-specified band cannot reach the whole range — fall back to
    automatic per-segment selection (max LO-edge margin across bands 1/2/3,
    tiebreaker = the qubits' current port band).
    """
    u = unit(coerce_to_integer=True)
    bands = {q.name: _qubit_band(q) for q in qubits}
    unique_bands = set(bands.values())
    if len(unique_bands) != 1:
        raise ValueError(
            f"All listed qubits must share the same readout port (and thus the "
            f"same currently-configured MW-FEM band). Got bands: {bands}"
        )
    current_band = next(iter(unique_bands))

    f_start_hz = node.parameters.freq_start_ghz * u.GHz
    f_stop_hz = node.parameters.freq_stop_ghz * u.GHz
    dz_hz = node.parameters.if_dead_zone_mhz * u.MHz
    if_max_hz = node.parameters.if_max_mhz * u.MHz

    user_band = node.parameters.band
    auto_priority = [current_band] + [b for b in (1, 2, 3) if b != current_band]
    if user_band is None:
        bands_priority = auto_priority
        band_mode = f"auto (priority {auto_priority})"
    elif can_band_cover(user_band, f_start_hz, f_stop_hz, dz_hz, if_max_hz):
        bands_priority = [user_band]
        band_mode = f"{user_band} (user-specified, covers full range)"
    else:
        bands_priority = auto_priority
        band_mode = (
            f"auto (priority {auto_priority}) — requested band {user_band} "
            f"cannot cover range within ±{if_max_hz/1e6:.0f} MHz IF, "
            f"falling back to per-segment selection"
        )

    segments = plan_segments(
        f_start_hz=f_start_hz,
        f_stop_hz=f_stop_hz,
        step_hz=node.parameters.frequency_step_in_mhz * u.MHz,
        bands_priority=bands_priority,
        if_dead_zone_hz=dz_hz,
        if_max_hz=if_max_hz,
    )
    node.namespace["band_mode"] = band_mode
    return segments


def _build_segment_program(node, qubits, seg, n_avg):
    """Build a per-segment QUA program scanning seg.dfs_hz once with n_avg shots.

    The uniform survey amplitude (``global_readout_amp``) is applied by overriding the probe's
    readout pulse amplitude in the config (see ``execute_qua_program``), so the QUA program just
    plays "readout" — no per-measure amplitude prefactor here.
    """
    u = unit(coerce_to_integer=True)
    num_qubits = len(qubits)
    with program() as prog:
        I = [declare(fixed) for _ in range(num_qubits)]
        Q = [declare(fixed) for _ in range(num_qubits)]
        I_st = [declare_stream() for _ in range(num_qubits)]
        Q_st = [declare_stream() for _ in range(num_qubits)]
        n = declare(int)
        n_st = declare_stream()
        df = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, seg.dfs_hz)):
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        rr.update_frequency(df)  # absolute IF; LO set in QUAM
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        rr.wait(rr.depletion_time * u.ns)
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(seg.n_points).average().save(f"I{i + 1}")
                Q_st[i].buffer(seg.n_points).average().save(f"Q{i + 1}")
    return prog


# %% {Plan run}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def plan_run(node: QualibrationNode[Parameters, Quam]):
    """Stash qubits, segments, and the global RF axis used by the analysis."""
    # Full list drives analysis/plotting/update_state (all qubit-name-keyed, feedline-agnostic).
    node.namespace["qubits"] = qubits = get_qubits(node)
    # The QUA program sweeps only ONE probe per feedline; non-probes are fanned out after
    # the fetch from their feedline-representative's trace (see execute_qua_program).
    reps, feedline_groups, rep_name_for_qubit = _feedline_representatives(node, qubits)
    node.namespace["qubits_reps"] = reps
    node.namespace["feedline_groups"] = feedline_groups
    node.namespace["rep_name_for_qubit"] = rep_name_for_qubit
    segments = _build_segments(node, reps)
    node.namespace["segments"] = segments

    node.log("======= WIDE SCAN PLAN =======")
    node.log(f"qubits: {[q.name for q in qubits]}")
    node.log(
        f"feedlines: {len(feedline_groups)} -> probes swept: {[r.name for r in reps]} "
        f"(other qubits fanned out from their feedline's trace)"
    )
    node.log(f"current port band: {_qubit_band(reps[0])}")
    node.log(f"band mode: {node.namespace.get('band_mode', '?')}")
    node.log(f"range: {node.parameters.freq_start_ghz:.3f} - {node.parameters.freq_stop_ghz:.3f} GHz")
    node.log(f"step: {node.parameters.frequency_step_in_mhz} MHz")
    for k, s in enumerate(segments):
        rf_lo = s.rf_hz.min() / 1e9
        rf_hi = s.rf_hz.max() / 1e9
        node.log(
            f"  seg{k}: band={s.band}, LO={s.lo_hz/1e9:.4f} GHz, "
            f"RF=[{rf_lo:.4f}, {rf_hi:.4f}] GHz, n_pts={s.n_points}"
        )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Simulate just the first segment's program (representative of the per-seg shape)."""
    qubits = node.namespace["qubits_reps"]
    segments = node.namespace["segments"]
    seg = segments[0]
    prog = _build_segment_program(node, qubits, seg, node.parameters.num_shots)
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    debug = False
    if debug:
        from pathlib import Path
        from qm import generate_qua_script
        file_name = Path(__file__).stem
        with open(Path(__file__).parent.parent / f"{file_name}_debug.py", 'w') as sourceFile:
            print(generate_qua_script(prog, config), file=sourceFile)
    samples, fig, wf_report = simulate_and_plot(qmm, config, prog, node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Per segment: tracked_updates LO, regenerate config, execute, fetch, revert.

    Only the per-feedline probe qubits are measured; after the fetch each measured trace
    is fanned out to every qubit on that feedline so downstream stays on the full list.
    """
    reps = node.namespace["qubits_reps"]
    qubits_all = node.namespace["qubits"]
    rep_name_for_qubit = node.namespace["rep_name_for_qubit"]
    segments = node.namespace["segments"]
    qmm = node.machine.connect()

    per_seg_datasets = []
    for seg_idx, seg in enumerate(segments):
        node.log(
            f"\n--- segment {seg_idx + 1}/{len(segments)}: "
            f"band={seg.band}, LO={seg.lo_hz/1e9:.4f} GHz ---"
        )
        with ExitStack() as stack:
            # Tracked-update the MW-FEM band first (so per-segment band
            # switches work for cross-band scans), then the upconverter and
            # downconverter LO. The band and the two LOs are independent
            # fields on the port objects; the QUAM comment about a single
            # reference propagating them is not generally true. Skipping a
            # write when the field is a QUAM reference is safe — the
            # reference will pick up the value from its source.
            for q in reps:
                q_upd = stack.enter_context(
                    tracked_updates(q, auto_revert=True, dont_assign_to_none=False)
                )
                _tracked_set_unless_reference(
                    q.resonator.opx_output, q_upd.resonator.opx_output,
                    "band", int(seg.band),
                )
                _tracked_set_unless_reference(
                    q.resonator.opx_input, q_upd.resonator.opx_input,
                    "band", int(seg.band),
                )
                _tracked_set_unless_reference(
                    q.resonator.opx_output, q_upd.resonator.opx_output,
                    "upconverter_frequency", int(seg.lo_hz),
                )
                _tracked_set_unless_reference(
                    q.resonator.opx_input, q_upd.resonator.opx_input,
                    "downconverter_frequency", int(seg.lo_hz),
                )

            # Uniform survey amplitude: override the probe's readout pulse amplitude directly
            # to global_readout_amp (the literal stored amplitude value, NOT a scale factor),
            # so the whole line is swept at that amplitude. Auto-reverts with the ExitStack
            # after the segment. Bounded only by the hardware DAC range (config validation),
            # not by the (-2, 2) QUA amplitude prefactor.
            if node.parameters.global_readout_amp is not None:
                for r in reps:
                    pulse_upd = stack.enter_context(
                        tracked_updates(
                            r.resonator.operations["readout"], auto_revert=True,
                            dont_assign_to_none=False,
                        )
                    )
                    pulse_upd.amplitude = float(node.parameters.global_readout_amp)

            # Static config IF validation uses each resonator's stored RF_frequency
            # against its (now-shared) upconverter LO. For a wide scan, stored RFs
            # can be hundreds of MHz away from the segment LO, blowing past the
            # 500 MHz hardware IF cap. Override every resonator's RF_frequency to
            # LO + safe IF so the config passes; runtime update_frequency(df)
            # drives the actual sweep regardless.
            u = unit(coerce_to_integer=True)
            safe_if_hz = int(node.parameters.if_dead_zone_mhz * u.MHz) + 1 * u.MHz
            for q in node.machine.qubits.values():
                lo = int(q.resonator.opx_output.upconverter_frequency)
                q_upd = stack.enter_context(
                    tracked_updates(q, auto_revert=True, dont_assign_to_none=False)
                )
                _tracked_set_unless_reference(
                    q.resonator, q_upd.resonator,
                    "RF_frequency", lo + safe_if_hz,
                )

            config = node.machine.generate_config()
            prog = _build_segment_program(node, reps, seg, node.parameters.num_shots)

            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(prog)
                seg_axes = {
                    "qubit": xr.DataArray([q.name for q in reps]),
                    "RF_frequency": xr.DataArray(
                        seg.rf_hz,
                        attrs={"long_name": "RF frequency", "units": "Hz"},
                    ),
                }
                data_fetcher = XarrayDataFetcher(job, seg_axes)
                for dataset in data_fetcher:
                    progress_counter(
                        data_fetcher.get("n", 0),
                        node.parameters.num_shots,
                        start_time=data_fetcher.t_start,
                    )
                node.log(job.execution_report())
            per_seg_datasets.append(dataset)

    # Concatenate across segments along the RF axis (qubit dim = the probe reps only)
    ds_combined = xr.concat(per_seg_datasets, dim="RF_frequency")
    ds_combined = ds_combined.sortby("RF_frequency")

    # Fan each feedline-probe trace out to every qubit on that feedline: the resonators of
    # all qubits on a line live in that single uniform sweep, so each qubit gets its line's
    # trace and is then fit near its own stored RF_frequency. Keeps the downstream
    # (analysis/plotting/update_state) keyed by qubit name over the full list, unchanged.
    ds_raw = xr.concat(
        [
            ds_combined.sel(qubit=rep_name_for_qubit[q.name], drop=True).expand_dims(qubit=[q.name])
            for q in qubits_all
        ],
        dim="qubit",
    )

    # Persist per-segment metadata so plotting (including the load-from-id path)
    # can do per-segment baseline normalization on the wide amplitude trace.
    ds_raw.attrs["segment_lo_hz"] = [int(s.lo_hz) for s in segments]
    ds_raw.attrs["segment_boundaries_hz"] = [
        int(round((segments[k].rf_hz.max() + segments[k + 1].rf_hz.min()) / 2))
        for k in range(len(segments) - 1)
    ]
    node.results["ds_raw"] = ds_raw

    # Pack the global sweep axes for downstream code symmetry with the pause/resume node
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray([q.name for q in qubits_all]),
        "RF_frequency": xr.DataArray(
            ds_raw.RF_frequency.values,
            attrs={"long_name": "RF frequency", "units": "Hz"},
        ),
    }


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset and rebuild namespace."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.parameters.re_fit_resonators = _stored_re_fit_resonators
    node.parameters.re_fit_centers_ghz = _stored_re_fit_centers_ghz
    node.parameters.re_fit_span_mhz = _stored_re_fit_span_mhz
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Find dip candidates, assign by proximity, refit narrow window per qubit."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot per-qubit wide traces with assignment overlays."""
    fig_raw_phase = plot_raw_phase(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    fig_fit_amplitude = plot_raw_amplitude_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    fig_local_amplitude = plot_local_amplitude_with_fit(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    fig_detrended_phase = plot_detrended_phase(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    fig_iq_circle = plot_iq_circle(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"]
    )
    plt.show()
    node.results["figures"] = {
        "phase": fig_raw_phase,
        "amplitude": fig_fit_amplitude,
        "amplitude_local": fig_local_amplitude,
        "detrended_phase": fig_detrended_phase,
        "iq_circle": fig_iq_circle,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write assigned resonator frequencies back to QUAM for successful qubits."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            q.resonator.f_01 = float(node.results["fit_results"][q.name]["frequency"])
            q.resonator.RF_frequency = float(node.results["fit_results"][q.name]["frequency"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results."""
    node.save()
