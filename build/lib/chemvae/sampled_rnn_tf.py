'''
sampled_rnn - based on rnn method in tensorflow_backend.py in keras

Main difference is in how to handle dimensions of states.


# think carefully about the distribution of the random sampled variables...

'''


def sampled_rnn(step_function, inputs, initial_states, units, random_seed,
                go_backwards=False, mask=None, rec_dp_constants=None,
                unroll=False, input_length=None):
    """Iterates over the time dimension of a tensor.
    # Arguments
        step_function: RNN step function.
            Parameters:
                input: tensor with shape `(samples, ...)` (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape `(samples, output_dim)`
                    (no time dimension).
                new_states: list of tensors, same length and shapes
                    as 'states'. The first state in the list must be the
                    output tensor at the previous timestep.
        inputs: tensor of temporal data of shape `(samples, time, ...)`
            (at least 3D).
        initial_states: tensor with shape (samples, output_dim)
            (no time dimension),
            containing the initial values for the states used in
            the step function.
        units : number of units in the output dimension.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape `(samples, time, 1)`,
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
        unroll: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
        input_length: not relevant in the TensorFlow implementation.
            Must be specified if using unrolling with Theano.

    # Returns
        A tuple, `(last_output, outputs, new_states)`.

            last_output: the latest output of the rnn, of shape `(samples, ...)`
            outputs: tensor with shape `(samples, time, ...)` where each
                etry `outputs[s, t]` is the output of the step function
                at time `t` for sample `s`.
            new_states: list of tensors, latest states returned by
                the step function, of shape `(samples, ...)`.

    # Raises
        ValueError: if input dimension is less than 3.
        ValueError: if `unroll` is `True` but input timestep is not a fixed number.
        ValueError: if `mask` is provided (not `None`) but states is not provided
            (`len(states)` == 0).
   """
    import numpy as np
    np.random.seed(random_seed)
    import tensorflow as tf
    tf.set_random_seed(random_seed)
    from tensorflow.python.ops import tensor_array_ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.framework import constant_op
    from tensorflow.python.framework import dtypes
    import keras.backend as K

    ndim = len(inputs.get_shape())
    if ndim < 3:
        raise ValueError('Input should be at least 3D.')

    if unroll == True:
        raise ValueError('Unrolling not implemented in sampled_rnn')
    if mask is not None:
        raise ValueError('Masking not implemented in sampled_rnn')

    # this switches dims to (time, samples, ...)
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))

    if go_backwards:
        inputs = reverse(inputs, 0)

    states = tuple(initial_states)

    # inputs shape: (time, samples, output_dims -- may be 3xoutput
    # because of cpu implementation)
    time_steps = tf.shape(inputs)[0]

    # Calculate one time step first
    # Generate a random cutoff probability for comparing to the cdf probility
    #   Generates one for each sample in the batch

    num_samples = tf.shape(inputs)[1]
    output_dim = int(initial_states[0].get_shape()[-1])
    random_cutoff_prob = tf.random_uniform(
        (num_samples,), minval=0., maxval=1.)

    # Ignore constants for the first run
    outputs, _ = step_function(inputs[0], {'initial_states': initial_states,
                                           'random_cutoff_prob': random_cutoff_prob,
                                           'rec_dp_mask': rec_dp_constants})

    output_ta = tensor_array_ops.TensorArray(
        dtype=outputs.dtype,
        size=time_steps,
        tensor_array_name='output_ta')
    input_ta = tensor_array_ops.TensorArray(
        dtype=inputs.dtype,
        size=time_steps,
        tensor_array_name='input_ta')
    input_ta = input_ta.unstack(inputs)
    time = tf.constant(0, dtype='int32', name='time')

    def _step(time, output_ta_t, *states):
        """RNN step function.

        # Arguments
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        # Returns
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        current_input = input_ta.read(time)
        random_cutoff_prob = tf.random_uniform(
            (num_samples,), minval=0, maxval=1)

        output, new_states = step_function(current_input,
                                           {'initial_states': states,
                                            'random_cutoff_prob': random_cutoff_prob,
                                            'rec_dp_mask': rec_dp_constants})
        # returned output is ( raw/sampled, batch, output_dim)
        axes = [1, 0] + list(range(2, K.ndim(output)))
        output = tf.transpose(output, (axes))
        for state, new_state in zip(states, new_states):
            new_state.set_shape(state.get_shape())
        output_ta_t = output_ta_t.write(time, output)
        return (time + 1, output_ta_t) + tuple(new_states)

    final_outputs = control_flow_ops.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_step,
        loop_vars=(time, output_ta) + states,
        parallel_iterations=1,
        swap_memory=True)

    last_time = final_outputs[0]
    output_ta = final_outputs[1]
    new_states = final_outputs[2:]

    # this is in order to get the second output in states (i.e. the sampled
    # output)
    outputs = output_ta.stack()[:, :, 1, :]
    last_output = output_ta.read(last_time - 1)[:, 1, :]

    # outputs is switched back to (samples, timesteps, output_dims)
    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states
