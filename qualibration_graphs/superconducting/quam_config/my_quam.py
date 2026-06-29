from dataclasses import field
from typing import Any, Dict, List, Optional

from quam.core import QuamBase, quam_dataclass
from quam_builder.architecture.superconducting.qpu import CavityQuam, FluxTunableQuam


@quam_dataclass
class TemporaryCalibrationData(QuamBase):
    """Transient per-qubit calibration state.

    Stored in ``Quam.temp_calibration[qubit_name]`` and persisted to the QUAM
    state so that values survive node boundaries within a calibration run.
    All fields are optional and default to None (absent until first written).
    """

    parameters: Dict[str, Any] = None

    # ── Spectroscopy adaptation ───────────────────────────────────────────────
    adaptive_frequency_span_mhz: Optional[float] = None
    adaptive_power_shift_dbm: Optional[float] = None
    adaptive_num_shots: Optional[int] = None

    # ── Blacklists ────────────────────────────────────────────────────────────
    blacklisted_qubit_points: Optional[List[List[float]]] = None
    blacklisted_resonator_frequencies: Optional[List[float]] = None

    # ── Resonator rollback values ─────────────────────────────────────────────
    initial_resonator_f01: Optional[float] = None
    initial_resonator_RF_frequency: Optional[float] = None

    # ── x180 / qubit rollback values ─────────────────────────────────────────
    initial_x180_amplitude: Optional[float] = None
    initial_qubit_f01: Optional[float] = None
    initial_rf_frequency: Optional[float] = None
    initial_x180_length_ns: Optional[float] = None

    # ── Spectroscopy result ───────────────────────────────────────────────────
    selected_power_dbm: Optional[float] = None
    selected_octave_gain_db: Optional[float] = None

    # ── Duration adaptation ───────────────────────────────────────────────────
    adaptive_x180_length_ns: Optional[float] = None

    # ── Metadata ─────────────────────────────────────────────────────────────
    last_updated: Optional[str] = None
    notes: Optional[str] = None


@quam_dataclass
class Quam(CavityQuam, FluxTunableQuam):
    """QUAM for flux-tunable transmons (with tunable couplers) coupled to SRF cavities.

    Adds `temp_calibration`: per-qubit temporary state used by adaptive calibration nodes.

    The load() override fixes a QUAM serialisation quirk where Octave loopbacks
    (Tuple[Tuple[str,str],str]) are round-tripped through JSON as nested lists,
    which typeguard rejects on reload.

    FluxTunableQuam.initialize_qpu already activates the TWPA(s) and sets qubit/coupler
    flux points, so no override is needed here.
    """

    temp_calibration: Dict[str, TemporaryCalibrationData] = None

    @classmethod
    def load(cls, filepath_or_dict=None, **kwargs) -> "Quam":
        """Load the QUAM state, patching Octave loopback tuples before validation.

        JSON has no tuple type, so QUAM serialises Octave loopbacks — which are
        typed as ``List[Tuple[Tuple[str, str], str]]`` — as nested lists.  On
        reload, typeguard rejects the inner list because it expects a
        ``Tuple[str, str]``, raising a validation error before the object is
        even constructed.

        This override loads the raw JSON dict first, converts the inner lists
        back to tuples, and only then passes the fixed dict to the standard QUAM
        instantiator via ``super().load()``.
        """
        if isinstance(filepath_or_dict, dict):
            contents = filepath_or_dict
        else:
            serialiser = cls.get_serialiser()
            contents, _ = serialiser.load(filepath_or_dict)

        # Fix loopbacks: JSON round-trip turns Tuple[str,str] -> list[str].
        # Convert back to tuple so typeguard validation passes.
        for oct_data in contents.get("octaves", {}).values():
            if isinstance(oct_data, dict):
                oct_data["loopbacks"] = [
                    (tuple(src) if isinstance(src, list) else src, dst)
                    for src, dst in oct_data.get("loopbacks", [])
                ]

        return super().load(contents, **kwargs)
