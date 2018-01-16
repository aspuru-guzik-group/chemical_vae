'''
A GRU layer capable of teacher forcing at train time, and sampling from a softmax at test time. 

# Example of use:
tgru = TerminalGRU(NCHARS, rnd_seed=42, return_sequences=True, activation='softmax', temperature=0.01, name='tgru_layer')


## Overview of code: main differences from regular GRU
- Takes in two inputs: 
    1. Previous layer of network (usually from a regular GRU) 
    2. True original sequence input (or a dummy if running in a test phase/free running)
- Sequential output at each time step is generated in one of two ways:
    1. Teacher forcing (train time):
         Uses true input (in original space) from previous time step to update the state for the current time step.
    2. Free running (test time)
         Uses sampled output from the previous time step to calculate the current input 
- State is passed as a dictionary containing the following. See step function below for details. 
    - 'initial_states',  'random_cutoff_prob',  'rec_dp_mask' 


    - a 'raw state' - calculated from the previous step (given by first dimension of state in step)
    - For train phase:
        - raw state is repeated twice
    - For test phase
        - a 'sampled state' - sampled using the raw state 
        - This then becomes the output for the layer at this timestep
- Uses 'sampled_rnn' in order to get around state changes

## Input shape:
    - list of 2 tensors, 
        1. 3D tensor (input from previous layer) : '(batch_size, timesteps, input_dim)'
        2. 3D tensor (true input sequence to model) : '(batch_size, timesteps, output_dim)'
        output_dim == units


## Output shape:
    - if 'return_sequences': 3D tensor with shape:
        '(batch_size, timesteps, output_dim)'
    - else: 2D tensor with shape '(batch_size, output_dim)'

## Other functions:
    -output_sampling(output, rand_matrix) : used to help sample the final one-hot output vector based on continuous output vector
            rand_matrix - Random matrix generated within sampled_rnn, (batch_size, )
    - get_initial_states(x)
        Generate random starting state (of zeros)


# Arguments:
    output_dim : dimensions of outputs at each time step, in each sample 
            called units in Keras 2.0 recurrent layer
    teacher_force_ratio : Ratio for using teacher forcing method vs. free_running method 
    temperature - Temperature for sampling  om tje GRU layer
    rnd_seed - a random seed to use (currently not being used)


This version of TerminalGRU is currrently implemented for self.implementation==0 only 
Other implementations will need to be ported over from original recurrent layer.

self.implementation ==2 : gpu
self.implementation ==1 : mem
self.implementation ==0 : cpu

'''
from keras.layers.recurrent import GRU
from keras import backend as K
from keras.engine import InputSpec
import numpy as np

if K.backend() == 'tensorflow':
    from .sampled_rnn_tf import sampled_rnn
else:
    raise NotImplemented("Backend not implemented")


class TerminalGRU(GRU):
    # Heavily adapted from GRU in recurrent.py
    # Implements professor forcing 

    def __init__(self, units,
                 temperature=1., rnd_seed=None, recurrent_dropout=0.0,
                 **kwargs):
        # @param: temperature - sampling temperature
        # Annealing will be handled in the callbacks
        super(TerminalGRU, self).__init__(units, **kwargs)
        self.units = units
        self.temperature = temperature
        self.rnd_seed = rnd_seed
        self.uses_learning_phase = True
        self.supports_masking = False
        self.units = units
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.input_spec = [InputSpec(ndim=3),
                           InputSpec(ndim=3)]

    def build(self, input_shape):
        # all of this is copied from GRU, except for one part commented below
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = [InputSpec(shape=(batch_size, None, self.input_dim)),
                           InputSpec(shape=(batch_size, None, self.units))]
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight((self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # adding an extra recurrent weight here, change from GRU layer:
        # this last recurrent weight applied to true sequence input from prev. timestep,
        #   or sampled output from prev. time step. 
        self.recurrent_kernel = self.add_weight(
            (self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.units * 4,),
                                        name='bias',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                  self.units:
                                  self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:self.units * 3]
        self.recurrent_kernel_y = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2: self.units * 3]
            self.bias_h = self.bias[self.units * 3:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

    def get_initial_states(self, x):
        # build an all-zero tensor of shape [(samples, output_dim), (samples, output_dim)]
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.random_uniform((self.input_dim, self.units))
        reducer = reducer / K.exp(reducer)

        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [K.stack([initial_state, initial_state]) for _ in range(len(self.states))]
        return initial_states

    def compute_mask(self, input, mask):
        # Forced to be single dimension, following behavior of Merge layer 
        # not implemented
        return None

    def get_constants(self, inputs, training=None):
        constants = []
        if 0. < self.recurrent_dropout < 1.:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('terminal gru runs on list of length 2')

        X = inputs[0]
        true_seq = inputs[1]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        # preprocessing makes input into right form for gpu/cpu settings
        # from original GRU code
        recurrent_dropout_constants = self.get_constants(X)[0]
        preprocessed_input = self.preprocess_input(X)

        #################
        ## Section for index matching of true inputs
        #################
        #  Basically, we need to add an extra timestep of just 0s for predicting the first timestep output

        axes = [1, 0] + list(range(2, K.ndim(true_seq)))

        true_seq = K.permute_dimensions(true_seq, axes)
        zeros = K.zeros_like(true_seq[:1, :, :])

        # add a column of zeros, remove last element
        true_seq = K.concatenate([zeros, true_seq[:K.int_shape(true_seq)[0] - 1, :, :]], axis=0)
        shifted_raw_inputs = K.permute_dimensions(true_seq, axes)

        ## concatenate to have same dimension as preprocessed inputs 3xoutput_dim
        # only for self.implementation = 0?
        shifted_raw_inputs = K.concatenate([shifted_raw_inputs,
                                            shifted_raw_inputs,
                                            shifted_raw_inputs], axis=2)

        all_inputs = K.stack([preprocessed_input, shifted_raw_inputs])
        num_dim = K.ndim(all_inputs)
        axes = [1, 2, 0] + list(range(3, num_dim))
        all_inputs = K.permute_dimensions(all_inputs, axes)

        # If not using true sequence, want to feed in a tensor of zeros instead.
        zeros_input_seq = K.zeros_like(preprocessed_input)
        test_phase_all_inputs = K.stack([preprocessed_input, zeros_input_seq])
        test_phase_all_inputs = K.permute_dimensions(test_phase_all_inputs, axes)

        all_inputs = K.in_train_phase(all_inputs, test_phase_all_inputs)

        last_output, outputs, states = sampled_rnn(self.step,
                                                   all_inputs,
                                                   initial_states,
                                                   self.units,
                                                   self.rnd_seed,
                                                   go_backwards=self.go_backwards,
                                                   rec_dp_constants=recurrent_dropout_constants,
                                                   mask=None)

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def compute_output_shape(self, input_shape):
        # expect input_shape is a list:
        assert type(input_shape) is list

        input_shapes = input_shape

        # from original recurrent unit, can probably delete entire if this works 
        if self.return_sequences:
            return input_shapes[1]
        else:
            return (input_shapes[1][0], input_shapes[1][1])

    def get_config(self):
        config = {'units': self.units,
                  'temperature': self.temperature,
                  'rnd_seed': self.rnd_seed}
        base_config = super(TerminalGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def output_sampling(self, output, rand_matrix):
        # Generates a sampled selection based on raw output state vector 
        # Creates a cdf vector and compares against a randomly generated vector
        # Requires a pre-generated rand_matrix (i.e. generated outside step function) 

        sampled_output = output / K.sum(output, axis=-1, keepdims=True)  # (batch_size, self.units)
        mod_sampled_output = sampled_output / K.exp(self.temperature)
        norm_exp_sampled_output = mod_sampled_output / K.sum(mod_sampled_output, axis=-1, keepdims=True)

        cdf_vector = K.cumsum(norm_exp_sampled_output, axis=-1)
        cdf_minus_vector = cdf_vector - norm_exp_sampled_output

        rand_matrix = K.stack([rand_matrix], axis=0)
        rand_matrix = K.stack([rand_matrix], axis=2)

        compared_greater_output = K.cast(K.greater(cdf_vector, rand_matrix), dtype='float32')
        compared_lesser_output = K.cast(K.less(cdf_minus_vector, rand_matrix), dtype='float32')

        final_output = compared_greater_output * compared_lesser_output
        return final_output

    def step(self, h, states):
        '''
        receives inputs for a time step
        @inp : h - [previous_layer_input, true_input_for_previous_timestep] at train time
               or  [previous_layer_input, zeros] at test time
        @inp : states - a dictionary, contains the following 
            - 'initial_states' - state vector
                 - At train time, this includes the true input sequence for the given time step, in addition to the state for the previous time step. 
                 - At test time,
            - 'random_cutoff_prob' - random cutoff matrix used for sampling at test time
            - 'rec_dp_mask' - for use with dropout (not tested - may break)

        @return: output - raw output, unsampled
        @return: final_output - output that has been sampled in test case

        '''

        ################
        # Parsing the states vector
        ################
        initial_states = states['initial_states']
        random_cutoff_vec = states['random_cutoff_prob']

        if self.recurrent_dropout > 0:
            rec_dp_mask = states['rec_dp_mask']
        else:
            rec_dp_mask = np.array([1., 1., 1., 1.], dtype='float32')

        h_tm1 = initial_states[0][:1, :, :]

        def teacher_forced(h, states):
            # switching from (batch_size, previous_layer_input|true_input, output_dim)
            #    to ( previous_layer_input|true_input, batch_size, output_dim)
            axes = [1, 0] + list(range(2, K.ndim(h)))
            h = K.permute_dimensions(h, axes)

            prev_layer_input = h[0:1, :, :]
            true_input = h[1:, :, :self.units]

            # this should correspond  to true input
            prev_sampled_output = true_input

            if self.implementation == 0:
                x_z = prev_layer_input[0, :, :self.units]
                x_r = prev_layer_input[0, :, self.units: 2 * self.units]
                x_h = prev_layer_input[0, :, 2 * self.units:]
            else:
                raise ValueError('Implementation type ' + self.implementation + ' is invalid')

            z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
                                                      self.recurrent_kernel_z))
            r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
                                                      self.recurrent_kernel_r))

            hh = self.activation(x_h +
                                 K.dot(r * h_tm1 * rec_dp_mask[2],
                                       self.recurrent_kernel_h) +
                                 K.dot(r * prev_sampled_output, self.recurrent_kernel_y))

            output = z * h_tm1 + (1. - z) * hh

            return K.stack([output, output])

        def free_running(h, states):

            prev_generated_output = initial_states[0][1:, :, :]
            prev_sampled_output = prev_generated_output

            # switching from (batch_size, previous_layer_input|true_input, output_dim)
            #    to ( previous_layer_input|true_input, batch_size, output_dim)
            axes = [1, 0] + list(range(2, K.ndim(h)))
            h = K.permute_dimensions(h, axes)

            prev_layer_input = h[0:1, :, :]

            if self.implementation == 0:
                x_z = prev_layer_input[0, :, :self.units]
                x_r = prev_layer_input[0, :, self.units: 2 * self.units]
                x_h = prev_layer_input[0, :, 2 * self.units:]

            z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
                                                      self.recurrent_kernel_z))
            r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
                                                      self.recurrent_kernel_r))

            hh = self.activation(x_h +
                                 K.dot(r * h_tm1 * rec_dp_mask[2],
                                       self.recurrent_kernel_h) +
                                 K.dot(r * prev_sampled_output, self.recurrent_kernel_y))

            output = z * h_tm1 + (1. - z) * hh

            final_output = self.output_sampling(output, random_cutoff_vec)

            return K.stack([output, final_output])

        output_2d_tensor = K.in_train_phase(teacher_forced(h, states),
                                            free_running(h, states))

        output_2d_tensor = K.squeeze(output_2d_tensor, 1)

        return output_2d_tensor, [output_2d_tensor]
