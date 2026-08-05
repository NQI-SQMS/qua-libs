def get_cr_op(node, qp):
    wf_type = node.parameters.wf_type
    return qp.cross_resonance.operations[wf_type]


def get_cr_duration(node, qp=None, with_x180=False, convert_values=None):
    """Calculate the total CR pulse duration based on the number of pulses.

    Args:
        node: The calibration node containing parameters.
        qp: The qubit pair object.
        with_x180: Boolean indicating if X180 pulses are included.

    Returns:
        Total CR pulse duration in nanoseconds.
    """
    # convert values if provided else use the duration from the waveform
    if convert_values is not None:
        gate_length = convert_values
    elif qp is not None:
        cr_op = get_cr_op(node, qp)
        gate_length = cr_op.length
    else:
        raise ValueError("Qubit pair (qp) or convert_values must be provided.")

    # adjust by echo and include x180 if specified
    cr_type = node.parameters.cr_type
    if "echo" in cr_type:
        if with_x180:
            if qp is None:
                raise ValueError("Qubit pair (qp) must be provided when with_x180 is True.")
            qc = qp.qubit_control
            x180_duration = qc.xy.operations["x180"].length
            gate_length += x180_duration
        gate_length *= 2

    return gate_length
