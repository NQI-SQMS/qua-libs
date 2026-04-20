from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters

from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters, IdleTimeNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    frequency_detuning_in_mhz: float = 1.0
    """Frequency detuning in MHz. Default is 1.0 MHz."""
    x180_operation: str = "x180"
    """Operation name used as the π pulse. The x90 half-pulse is derived by playing this
    operation at amplitude_scale=0.5. Default is 'x180'."""
    selective_state_update: bool = False
    """If True and x180_operation != 'x180', store the frequency correction in the chosen
    operation's detuning field instead of updating q.f_01 / q.xy.RF_frequency / q.T2ramsey.
    Has no effect when x180_operation == 'x180'."""
    correct_with_pulse_detuning: bool = False
    """If True, shift the qubit XY element IF by the chosen pulse's detuning value before
    playing the Ramsey sequence and restore it afterwards. Only applies to DragGaussianPulse
    operations that have a detuning field."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    IdleTimeNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
