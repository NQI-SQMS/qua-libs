# %% {Imports}
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *
from qm.qua.lib import Cast

from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter

from qualibrate import QualibrationNode
from calibration_utils.shared import (
    _get_cavity_mode,
    _get_pair_components,
    apply_confusion_matrix_correction,
)
from calibration_utils.wigner_tomography_2d import (
    Parameters,
    resolve_probe_displacements,
    resolve_parity_time_ns,
    reconstruct_state,
    compute_fidelity,
    plot_wigner_disps,
    plot_wigner_result,
    plot_density_matrix,
    build_raw_dataset,
    dataset_to_arrays,
)
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

logger = logging.getLogger(__name__)


# %% {Node initialisation}
description = """
        2D WIGNER TOMOGRAPHY (36)

Measures the Wigner function W(β) of a prepared cavity Fock state using
optimised probe displacements that minimise the tomography matrix condition
number κ, reducing the number of measurements by ~10–30× vs. a square grid
while improving reconstruction accuracy.

Sequence (repeated for polarity = +1 and −1):
  For each probe point β_k (N_probe points):
    1. Thermalize cavity (T1 × factor) and reset qubit.
    2. Prepare Fock |n⟩ in the cavity:
         'sideband'         — ladder of f{j}g{j+1} sideband π-pulses
         'snap_displacement'— sequential Displacement–SNAP protocol
    3. Probe displacement D(−β_k) applied to cavity.
    4. Parity Ramsey (strict timing):
         x90 → wait(t_parity) → ±x90   (polarity determines sign)
    5. Qubit readout (state discrimination).

Analysis:
  parity[k] = P_e(polarity=−1)[k] − P_e(polarity=+1)[k]
  ρ = reconstruct_state(probe_disps, parity, N_ph)  [lstsq + PSD projection]

Probe displacements are loaded from probe_displacements_path when available,
otherwise optimised (L-BFGS-B) and cached. Set probe_displacements_path to the
path printed in the node log after the first run to skip re-optimisation.

Prerequisites:
  - Node 28: parity_time_ns (or chi_hz in QuAM)
  - Node 30: displacement_alpha_max in CavityTransmonPair
  - Sideband prep: nodes 26–26h (fNgN1_* chain, sideband_drive in QuAM)
  - SNAP prep: selective_x180 operation on qubit.xy; snap_displacement_photons
               set in parameters (list of n+1 amplitudes in photon units)
"""

node = QualibrationNode[Parameters, Quam](
    name="36_wigner_tomography_2d",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.mode_name = "alice"
    # node.parameters.num_shots = 400
    # node.parameters.n_fock_cutoff = 4
    # node.parameters.target_fock_level = 1
    # node.parameters.fock_prep_method = "sideband"
    # node.parameters.probe_displacements_path = None   # set after first run
    # node.parameters.parity_time_ns = 7000
    pass


node.machine = Quam.load()


# %% {Compute probe displacements}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def compute_probe_displacements(node: QualibrationNode[Parameters, Quam]):
    """Load or optimise probe displacements and store in namespace."""
    sm = getattr(node, "storage_manager", None)
    root = getattr(sm, "root_data_folder", None)
    results_dir = Path(root) if root else Path(".")
    disps, kappa, npz_path = resolve_probe_displacements(node, results_dir)
    # Store as absolute path so it can be found when loading from a different cwd
    npz_path_abs = str(Path(npz_path).resolve())
    node.namespace["probe_disps"] = disps
    node.namespace["kappa"] = kappa
    node.results["probe_displacements_path"] = npz_path_abs
    node.results["probe_displacements_re"] = disps.real
    node.results["probe_displacements_im"] = disps.imag
    node.results["kappa"] = kappa
    node.log(f"Probe displacements: {len(disps)} pts, κ={kappa:.3f}, saved/loaded from '{npz_path_abs}'")


# %% {Create QUA program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build QUA programs (polarity +1 and −1) and stash shared parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Expect a single qubit for Wigner tomography
    qubit_list = list(qubits)
    if len(qubit_list) != 1:
        raise ValueError(
            f"36_wigner_tomography_2d expects exactly one qubit, got {len(qubit_list)}."
        )
    qubit = qubit_list[0]
    node.namespace["qubit"] = qubit

    pair, _, sb_drive, cavity_mode = _get_pair_components(node)
    node.namespace["cavity_mode"] = cavity_mode

    chi_hz = float(pair.chi) if getattr(pair, "chi", None) is not None else 0.0

    displaced_threshold = None
    if node.parameters.use_state_discrimination and node.parameters.use_displaced_threshold:
        _t = getattr(pair, "ge_iq_threshold_displaced", None)
        if _t is not None:
            displaced_threshold = float(_t)

    parity_time_ns = resolve_parity_time_ns(node)
    node.namespace["parity_time_ns"] = parity_time_ns
    parity_clk = parity_time_ns // 4  # QUA clock cycles (4 ns each)

    t1_s = getattr(cavity_mode, "T1", 100e-6)
    factor = getattr(cavity_mode, "thermalization_time_factor", 5.0)
    therm_clk = int(np.clip(t1_s * factor * 1e9 / 4, 4, 2_500_000_000))

    alpha_max = float(getattr(pair, "displacement_alpha_max", 1.0))
    node.namespace["alpha_max"] = alpha_max

    probe_disps = node.namespace["probe_disps"]
    N_probe = len(probe_disps)
    node.namespace["N_probe"] = N_probe

    # Probe displacements in amplitude-scale units, negated for D(−β) convention
    probe_re = (-probe_disps.real / alpha_max).tolist()
    probe_im = (-probe_disps.imag / alpha_max).tolist()

    prep_method = node.parameters.fock_prep_method
    n_fock = node.parameters.target_fock_level
    protocol = node.parameters.fock_prep_protocol if prep_method == "sideband" else "sfo"
    ff_repeat = node.parameters.sfp_ff_repeat
    sfp_max_retries = node.parameters.sfp_max_retries
    pf_max_retries = node.parameters.pf_max_retries
    # Parity expected after the parity Ramsey for the target Fock level
    expected_parity = "odd" if (n_fock % 2 == 1) else "even"
    ge_readout_threshold = float(qubit.resonator.operations["readout"].threshold)

    # D-SNAP pre-computation (Python time)
    displacement_scales = None
    snap_qubit_ifs = None
    if prep_method == "snap_displacement":
        photons = node.parameters.snap_displacement_photons
        if photons is None or len(photons) != n_fock + 1:
            raise ValueError(
                f"snap_displacement_photons must be a list of {n_fock + 1} values "
                f"(for target_fock_level={n_fock}). Got: {photons}"
            )
        _AMP_MAX = 2.0 - 2 ** -16
        displacement_scales = []
        for k, ph in enumerate(photons):
            s = ph / alpha_max
            if abs(s) > _AMP_MAX:
                raise ValueError(
                    f"snap_displacement_photons[{k}]={ph} / alpha_max={alpha_max} "
                    f"= {s:.4f} exceeds the QUA hardware limit ±{_AMP_MAX:.6f}."
                )
            displacement_scales.append(float(s))
        snap_qubit_ifs = [pair.ge_if_at_fock(qubit, k) for k in range(n_fock)]

    n_avg = node.parameters.num_shots
    cav_drive_name = cavity_mode.cavity_mode_drive.name

    def _do_cavity_reset():
        """Emit QUA cavity reset instructions (inline helper, avoids duplication)."""
        if node.parameters.cavity_reset_type == "active_sideband":
            cavity_mode.reset(
                "active_sideband",
                node.parameters.simulate,
                log_callable=node.log,
                sideband_drive=sb_drive,
                qubit_thermalization_time=qubit.thermalization_time,
                fock_n=node.parameters.cavity_active_cooling_fock_n,
                sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                chi_hz=chi_hz,
                pair=pair,
            )
        elif node.parameters.cavity_reset_type == "active_sideband_v2":
            cavity_mode.reset(
                "active_sideband_v2",
                node.parameters.simulate,
                log_callable=node.log,
                sideband_drive=sb_drive,
                qubit=qubit,
                qubit_thermalization_time=qubit.thermalization_time,
                fock_n=node.parameters.cavity_active_cooling_fock_n,
                sideband_pulse_duration_ns=node.parameters.sideband_pulse_duration_ns,
                chi_hz=chi_hz,
                pair=pair,
                n_repeats=node.parameters.cavity_active_cooling_n_repeats,
            )
        else:
            cavity_mode.cavity_mode_drive.wait(therm_clk)

    def _do_qubit_reset():
        """Emit QUA qubit reset instructions.

        For SFP/SFP+PF protocols, uses reset_qubit_active_gef() which handles
        leakage to |f⟩ — critical because sideband operations can strand the
        qubit outside the ge subspace.  Falls back to standard reset for SFO.
        """
        if protocol != "sfo" and node.parameters.use_active_gef_qubit_reset:
            qubit.reset_qubit_active_gef()
        else:
            qubit.reset(
                node.parameters.reset_type,
                node.parameters.simulate,
                log_callable=node.log,
            )

    def _build_program(polarity: int):
        with program() as prog:
            n = declare(int)
            idx = declare(int)
            a_re = declare(fixed)
            a_im = declare(fixed)
            probe_re_arr = declare(fixed, value=probe_re)
            probe_im_arr = declare(fixed, value=probe_im)

            I = declare(fixed)
            Q = declare(fixed)
            state = declare(int)
            state_st = declare_stream()
            n_st = declare_stream()

            # Extra variables for SFP+PF parity filter loop
            if protocol == "sfp_pf":
                pf_accepted = declare(bool)
                retry_count = declare(int)
                I_pf = declare(fixed)

            node.machine.initialize_qpu(target=qubit)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(idx, 0, idx < N_probe, idx + 1):
                    assign(a_re, probe_re_arr[idx])
                    assign(a_im, probe_im_arr[idx])

                    # ----------------------------------------------------------------
                    # Steps 1+2: Reset and Fock state preparation
                    # ----------------------------------------------------------------
                    if protocol == "sfp_pf":
                        # Parity-filter loop: handles its own reset on each retry
                        assign(pf_accepted, False)
                        assign(retry_count, 0)
                        with while_(~pf_accepted & (retry_count < pf_max_retries)):
                            _do_cavity_reset()
                            _do_qubit_reset()
                            align()
                            if n_fock > 0:
                                pair.fock_prep_qua_sfp(n_fock, qubit, ff_repeat, sfp_max_retries)
                            align()
                            qubit.xy.update_frequency(pair.ge_if_at_fock(qubit, 0))
                            pair.parity_filter_qua(qubit, expected_parity, I_pf, pf_accepted)
                            assign(retry_count, retry_count + 1)

                        # After the PF loop the qubit is projected; reset it to |g⟩
                        # using the last measurement result — no extra measurement needed.
                        qubit.xy.update_frequency(pair.ge_if_at_fock(qubit, 0))
                        with if_(I_pf > ge_readout_threshold):
                            qubit.xy.play("x180")
                        align()

                    else:
                        # SFO or SFP: outer reset then prep
                        _do_cavity_reset()
                        _do_qubit_reset()
                        align()

                        if prep_method == "sideband":
                            if n_fock > 0:
                                if protocol == "sfp":
                                    pair.fock_prep_qua_sfp(n_fock, qubit, ff_repeat, sfp_max_retries)
                                else:  # sfo
                                    pair.fock_prep_qua(n_fock, qubit)
                            align()
                            qubit.xy.update_frequency(pair.ge_if_at_fock(qubit, 0))

                        else:  # snap_displacement
                            for k in range(n_fock):
                                cavity_mode.cavity_mode_drive.play(
                                    "displacement", amplitude_scale=displacement_scales[k]
                                )
                                align(qubit.xy.name, cav_drive_name)
                                qubit.xy.update_frequency(snap_qubit_ifs[k])
                                with strict_timing_():
                                    qubit.xy.play("selective_x180")
                                    qubit.xy.play("selective_x180")
                                align(qubit.xy.name, cav_drive_name)
                            cavity_mode.cavity_mode_drive.play(
                                "displacement", amplitude_scale=displacement_scales[n_fock]
                            )
                            align()
                            qubit.xy.update_frequency(pair.ge_if_at_fock(qubit, 0))

                    # 3. Probe displacement D(−β_k)
                    align(cav_drive_name, qubit.xy.name)
                    play("displacement" * amp(a_re, 0, a_im, 0), cav_drive_name)

                    # 4. Parity Ramsey (strict timing)
                    align(cav_drive_name, qubit.xy.name)
                    with strict_timing_():
                        qubit.xy.play("x90")
                        qubit.xy.wait(parity_clk)
                        qubit.xy.play("x90", amplitude_scale=polarity)

                    # 5. Readout
                    align()
                    _thresh = (displaced_threshold if displaced_threshold is not None else None) if node.parameters.use_state_discrimination else 0
                    qubit.readout_state(state, I=I, Q=Q, state_st=state_st, threshold=_thresh)

            with stream_processing():
                n_st.save("n")
                state_st.buffer(N_probe).average().save("state")

        return prog

    node.namespace["qua_program_plus"] = _build_program(polarity=+1)
    node.namespace["qua_program_minus"] = _build_program(polarity=-1)
    node.log(
        f"QUA programs built — N_probe={N_probe}, parity_time={parity_time_ns} ns, "
        f"prep='{prep_method}', protocol='{protocol}', Fock |{n_fock}⟩"
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program_plus"], node.parameters
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_programs(node: QualibrationNode[Parameters, Quam]):
    """Run both polarity programs and store raw excitation probabilities."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    n_avg = node.parameters.num_shots

    def _run_polarity(prog, label: str) -> np.ndarray:
        node.log(f"Executing Wigner program (polarity {label}) …")
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(prog)
            results_obj = fetching_tool(job, data_list=["n", "state"], mode="live")
            n_curr = 0
            state_arr = np.zeros(node.namespace["N_probe"])
            while results_obj.is_processing():
                n_curr, state_arr = results_obj.fetch_all()
                progress_counter(n_curr, n_avg, start_time=results_obj.get_start_time())
            n_curr, state_arr = results_obj.fetch_all()
            node.log(job.execution_report())
        return np.asarray(state_arr)

    state_plus = _run_polarity(node.namespace["qua_program_plus"], "+1")
    state_minus = _run_polarity(node.namespace["qua_program_minus"], "-1")

    node.results["state_plus"] = state_plus.tolist()
    node.results["state_minus"] = state_minus.tolist()


# %% {Load historical data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id

    # Prefer the self-contained dataset (has probe displacements embedded)
    ds = node.results.get("ds_raw")
    if ds is not None:
        probe_disps, _, _, _ = dataset_to_arrays(ds)
        node.namespace["probe_disps"] = probe_disps
        node.namespace["kappa"] = float(ds.attrs.get("kappa", float("nan")))
        node.log(f"Restored {len(probe_disps)} probe displacements from ds_raw.")
        return

    # Fallback: try the .npz file (may fail if path is stale/relative)
    npz_path = node.results.get("probe_displacements_path")
    if npz_path and Path(npz_path).exists():
        data = np.load(npz_path)
        node.namespace["probe_disps"] = data["disps"]
        node.namespace["kappa"] = float(data["kappa"])
        node.log(f"Restored probe displacements from npz: {npz_path}")
    else:
        node.log(
            "WARNING: could not restore probe displacements — "
            "ds_raw not found and probe_displacements_path is missing or stale."
        )


# %% {Analyse data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    state_plus = np.asarray(node.results["state_plus"])
    state_minus = np.asarray(node.results["state_minus"])

    if node.parameters.use_confusion_matrix_correction:
        qubit = node.namespace.get("qubit")
        if qubit is not None:
            cm = getattr(getattr(qubit, "resonator", None), "confusion_matrix", None)
            if cm is not None:
                cm_inv = np.linalg.inv(np.array(cm))
                for arr in [state_plus, state_minus]:
                    p = np.stack([1.0 - arr, arr], axis=-1)
                    p_corr = (cm_inv @ p.T).T
                    arr[:] = p_corr[..., 1]

    parity = state_plus - state_minus
    N_ph = node.parameters.n_fock_cutoff
    probe_disps = node.namespace["probe_disps"]

    rho = reconstruct_state(probe_disps, parity, N_ph)
    node.namespace["rho"] = rho
    node.results["rho_real"] = np.real(rho).tolist()
    node.results["rho_imag"] = np.imag(rho).tolist()
    node.results["parity"] = parity.tolist()
    node.results["parity_time_ns"] = node.namespace.get("parity_time_ns")
    node.results["n_fock_cutoff"] = N_ph
    node.results["target_fock_level"] = node.parameters.target_fock_level

    # Save raw dataset (self-contained: embeds probe displacements so the
    # experiment can be re-analysed / re-plotted without the .npz file)
    node.results["ds_raw"] = build_raw_dataset(
        probe_disps,
        state_plus,
        state_minus,
        parity,
        n_fock_cutoff=N_ph,
        target_fock_level=node.parameters.target_fock_level,
        parity_time_ns=node.namespace.get("parity_time_ns"),
        mode_name=node.parameters.mode_name,
        kappa=node.namespace.get("kappa", float("nan")),
    )

    target_fock = node.parameters.target_fock_level
    psi_target = np.zeros(N_ph, dtype=complex)
    psi_target[min(target_fock, N_ph - 1)] = 1.0
    fidelity = compute_fidelity(rho, psi_target)
    node.results["fidelity"] = float(fidelity)

    pn = np.real(np.diag(rho))
    node.log(
        f"Reconstruction done  (N_ph={N_ph}, "
        f"Tr[ρ]={float(np.trace(rho).real):.4f}, "
        f"F|{target_fock}⟩={fidelity*100:.2f}%, "
        f"P(n=0..{N_ph-1})={np.round(pn, 3).tolist()})"
    )
    node.outcomes = {"reconstruction": "successful"}


# %% {Plot data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    rho = node.namespace.get("rho")
    if rho is None:
        rho = (
            np.array(node.results["rho_real"])
            + 1j * np.array(node.results["rho_imag"])
        )

    probe_disps = node.namespace.get("probe_disps")
    kappa = node.namespace.get("kappa", float("nan"))
    N_ph = node.parameters.n_fock_cutoff
    N_exp = len(probe_disps)
    wigner_range = node.parameters.wigner_range if node.parameters.wigner_range > 0 else None

    parity = np.asarray(node.results["parity"]) if "parity" in node.results else None
    fidelity = node.results.get("fidelity")

    _prep_label = node.parameters.fock_prep_method
    if node.parameters.fock_prep_method == "sideband":
        _prep_label = f"{node.parameters.fock_prep_method}/{node.parameters.fock_prep_protocol}"
    title_str = (
        f"mode={node.parameters.mode_name}, "
        f"|{node.parameters.target_fock_level}⟩ via {_prep_label}"
    )

    fig_disps = plot_wigner_disps(probe_disps, kappa, N_ph, N_exp, title=title_str)
    fig_result = plot_wigner_result(
        rho, probe_disps, kappa,
        parity=parity,
        fidelity=fidelity,
        wigner_range=wigner_range,
        title=title_str,
        n_grid=node.parameters.n_grid,
    )
    fig_rho = plot_density_matrix(rho, title=f"Density matrix — {title_str}", fidelity=fidelity)

    plt.show()
    node.results["figures"] = {
        "probe_displacements": fig_disps,
        "wigner_result": fig_result,
        "density_matrix": fig_rho,
    }


# %% {Save results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
