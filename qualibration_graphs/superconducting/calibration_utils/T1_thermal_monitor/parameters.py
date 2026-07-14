from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 200
    """Number of averages per sweep (shared by T1, Ramsey, echo, and RPM sweeps)."""
    n_iter: int = 60
    """Number of (T1 + Ramsey + echo + RPM) iterations to repeat."""

    # ── Ramsey (T2* + qubit frequency) sweep ────────────────────────────────
    x180_operation: str = "x180"
    """Operation name used as the π pulse. The x90 half-pulse is derived by playing this
    operation at amplitude_scale=0.5. Default is 'x180'."""
    correct_with_pulse_detuning: bool = False
    """If True, shift the qubit XY element IF by the chosen pulse's detuning value before
    playing the Ramsey sequence and restore it afterwards."""
    ramsey_min_wait_time_in_ns: int = 16
    """Minimum Ramsey idle time [ns]. Default is 16 ns."""
    ramsey_max_wait_time_in_ns: int = 15000
    """Maximum Ramsey idle time [ns]. Default is 15 µs."""
    ramsey_wait_time_num_points: int = 50
    """Number of points in the Ramsey idle-time sweep. Default is 50."""
    ramsey_log_or_linear_sweep: str = "log"
    """Spacing of the Ramsey idle-time sweep: 'log' (geomspace) or 'linear' (linspace). Default is 'log'."""
    ramsey_frequency_detuning_in_mhz: float = 1.0
    """Virtual detuning applied via frame rotation for the Ramsey oscillation [MHz]. Default is 1 MHz."""

    # ── T2 echo sweep ────────────────────────────────────────────────────────
    echo_min_wait_time_in_ns: int = 100
    """Minimum echo half-idle time [ns]. Default is 100 ns."""
    echo_max_wait_time_in_ns: int = 150000
    """Maximum echo half-idle time [ns]. Default is 150 µs."""
    echo_wait_time_num_points: int = 50
    """Number of points in the echo idle-time sweep. Default is 50."""
    echo_log_or_linear_sweep: str = "log"
    """Spacing of the echo idle-time sweep: 'log' (geomspace) or 'linear' (linspace). Default is 'log'."""

    # ── RPM sweep ────────────────────────────────────────────────────────────
    min_amp_factor: float = 0.0
    """Minimum EF amplitude scale factor for the RPM sweep."""
    max_amp_factor: float = 2.0
    """Maximum EF amplitude scale factor for the RPM sweep."""
    amp_factor_step: float = 0.05
    """Step size for the RPM amplitude scale sweep."""
    ef_operation: str = "EF_x180"
    """Name of the EF π-pulse operation on qubit.xy."""
    use_confusion_matrix_correction: bool = False
    """Apply ge confusion matrix correction to averaged state probabilities."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
