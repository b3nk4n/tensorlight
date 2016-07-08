from __future__ import absolute_import
from __future__ import division

import collections
import six

import numpy as np
import tensorflow as tf
import tensortools as tt

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


class BasicLSTMConv2DCell(tf.nn.rnn_cell.RNNCell):
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                 state_is_tuple=False, activation=tanh):
        """Initialize the basic LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (see above).
            input_size: Deprecated and unused.
            state_is_tuple: If True, accepted and returned states are 2-tuples of
                the `c_state` and `m_state`.  By default (False), they are concatenated
                along the column axis.  This default behavior will soon be deprecated.
            activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            print(
                "Warning! %s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True." % self)
        if input_size is not None:
            print("Warning! %s: The input_size parameter is deprecated." % self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(1, 2, state)
                
            concat = _linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(1, 4, concat)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                 self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat(1, [new_c, new_h])
            return new_h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (_is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not _is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.constant_initializer(bias_start))
    return res + bias_term


def _is_sequence(seq):
    return (isinstance(seq, collections.Sequence)
            and not isinstance(seq, six.string_types))









"""def _get_session():
    return tf.get_default_session()


def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None):
    '''Iterates over the time dimension of a tensor.
    # Arguments
        inputs: tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape (samples, ...) (no time dimension),
                new_states: list of tensors, same length and shapes
                    as 'states'.
        initial_states: tensor with shape (samples, ...) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration over
            the time dimension in reverse order.
        mask: binary tensor with shape (samples, time, 1),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
    # Returns
        A tuple (last_output, outputs, new_states).
            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
    '''
    ndim = len(inputs.get_shape())
    assert ndim >= 3, "Input should be at least 3D."
    axes = [1, 0] + list(range(2, ndim))
    inputs = tf.transpose(inputs, (axes))
    input_list = tf.unpack(inputs)
    if constants is None:
        constants = []

    states = initial_states
    successive_states = []
    successive_outputs = []
    if go_backwards:
        input_list.reverse()

    if mask is not None:
        # Transpose not supported by bool tensor types, hence round-trip to uint8.
        mask = tf.cast(mask, tf.uint8)
        if len(mask.get_shape()) == ndim-1:
            mask = expand_dims(mask)
        mask = tf.cast(tf.transpose(mask, axes), tf.bool)
        mask_list = tf.unpack(mask)

        for input, mask_t in zip(input_list, mask_list):
            output, new_states = step_function(input, states + constants)

            # tf.select needs its condition tensor to be the same shape as its two
            # result tensors, but in our case the condition (mask) tensor is
            # (nsamples, 1), and A and B are (nsamples, ndimensions). So we need to
            # broadcast the mask to match the shape of A and B. That's what the
            # tile call does, is just repeat the mask along its second dimension
            # ndimensions times.
            tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(output)[1]]))

            if len(successive_outputs) == 0:
                prev_output = zeros_like(output)
            else:
                prev_output = successive_outputs[-1]

            output = tf.select(tiled_mask_t, output, prev_output)

            return_states = []
            for state, new_state in zip(states, new_states):
                # (see earlier comment for tile explanation)
                tiled_mask_t = tf.tile(mask_t, tf.pack([1, tf.shape(new_state)[1]]))
                return_states.append(tf.select(tiled_mask_t, new_state, state))

            states = return_states
            successive_outputs.append(output)
            successive_states.append(states)
    else:
        for input in input_list:
            output, states = step_function(input, states + constants)
            successive_outputs.append(output)
            successive_states.append(states)

    last_output = successive_outputs[-1]
    outputs = tf.pack(successive_outputs)
    new_states = successive_states[-1]

    axes = [1, 0] + list(range(2, len(outputs.get_shape())))
    outputs = tf.transpose(outputs, axes)
    return last_output, outputs, new_states


def conv2d(x, kernel, strides=(1, 1), border_mode='SAME',
           image_shape=None, filter_shape=None):
    '''Runs on cuDNN if available.
    # Arguments
        border_mode: string, "same" or "valid".
    '''
    strides = (1,) + strides + (1,)

    x = tf.nn.conv2d(x, kernel, strides, padding=border_mode)

    return x



class Layer(object):
    '''Abstract base layer class.
    All Keras layers accept certain keyword arguments:
        trainable: boolean. Set to "False" before model compilation
            to freeze layer weights (they won't be updated further
            during training).
        input_shape: a tuple of integers specifying the expected shape
            of the input samples. Does not includes the batch size.
            (e.g. `(100,)` for 100-dimensional inputs).
        batch_input_shape: a tuple of integers specifying the expected
            shape of a batch of input samples. Includes the batch size
            (e.g. `(32, 100)` for a batch of 32 100-dimensional inputs).
    '''
    def __init__(self, **kwargs):
        if not hasattr(self, 'trainable_weights'):
            self.trainable_weights = []
        if not hasattr(self, 'non_trainable_weights'):
            self.non_trainable_weights = []

        allowed_kwargs = {'input_shape',
                          'trainable',
                          'batch_input_shape',
                          'cache_enabled',
                          'name'}
        for kwarg in kwargs:
            assert kwarg in allowed_kwargs, 'Keyword argument not understood: ' + kwarg
        if 'batch_input_shape' in kwargs:
            self.set_input_shape(tuple(kwargs['batch_input_shape']))
        elif 'input_shape' in kwargs:
            self.set_input_shape((None,) + tuple(kwargs['input_shape']))
        self.trainable = True
        if 'trainable' in kwargs:
            self.trainable = kwargs['trainable']
        self.name = self.__class__.__name__.lower()
        if 'name' in kwargs:
            self.name = kwargs['name']
        self.cache_enabled = True
        if 'cache_enabled' in kwargs:
            self.cache_enabled = kwargs['cache_enabled']

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def cache_enabled(self):
        return self._cache_enabled

    @cache_enabled.setter
    def cache_enabled(self, value):
        self._cache_enabled = value

    def __call__(self, X, mask=None, train=False):
        # turn off layer cache temporarily
        tmp_cache_enabled = self.cache_enabled
        self.cache_enabled = False
        # create a temporary layer
        layer = Layer(batch_input_shape=self.input_shape)
        layer.name = "dummy"
        layer.input = X
        if hasattr(self, 'get_input_mask'):
            layer.get_input_mask = lambda _: mask 
        # set temporary previous
        tmp_previous = None
        if hasattr(self, 'previous'):
            tmp_previous = self.previous
        self.set_previous(layer, False)
        Y = self.get_output(train=train)
        # return previous to what it was
        if tmp_previous is not None:
            self.set_previous(tmp_previous, False)
        else:
            self.clear_previous(False)
        self.cache_enabled = tmp_cache_enabled
        return Y

    def set_previous(self, layer, reset_weights=True):
        '''Connect a layer to its parent in the computational graph.
        '''
        assert self.nb_input == layer.nb_output == 1, 'Cannot connect layers: input count and output count should be 1.'
        if hasattr(self, 'input_ndim'):
            assert self.input_ndim == len(layer.output_shape), ('Incompatible shapes: layer expected input with ndim=' +
                                                                str(self.input_ndim) +
                                                                ' but previous layer has output_shape ' +
                                                                str(layer.output_shape))
        if layer.get_output_mask() is not None:
            assert self.supports_masked_input(), 'Cannot connect non-masking layer to layer with masked output.'
        if not reset_weights:
            assert layer.output_shape == self.input_shape, ('Cannot connect layers without resetting weights: ' + 
                                                            'expected input with shape ' +
                                                            str(self.input_shape) +
                                                            ' but previous layer has output_shape ' +
                                                            str(layer.output_shape))
        self.previous = layer
        if reset_weights:
            self.build()

    def clear_previous(self, reset_weights=True):
        '''Unlink a layer from its parent in the computational graph.
        This is only allowed if the layer has an `input` attribute.
        '''
        if not hasattr(self, 'input'):
            raise Exception('Cannot clear previous for non-input layers')
        if hasattr(self, 'previous'):
            del self.previous
            if reset_weights:
                self.build()

    def build(self):
        '''Instantiation of layer weights.
        Called after `set_previous`, or after `set_input_shape`,
        once the layer has a defined input shape.
        Must be implemented on all layers that have weights.
        '''
        pass

    @property
    def trainable(self):
        if hasattr(self, '_trainable'):
            return self._trainable
        else:
            return True

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def nb_input(self):
        return 1

    @property
    def nb_output(self):
        return 1

    @property
    def input_shape(self):
        # if layer is not connected (e.g. input layer),
        # input shape can be set manually via _input_shape attribute.
        if hasattr(self, 'previous'):
            if hasattr(self, 'shape_cache') and self.cache_enabled:
                previous_layer_id = id(self.previous)
                if previous_layer_id in self.shape_cache:
                    return self.shape_cache[previous_layer_id]
            previous_size = self.previous.output_shape
            if hasattr(self, 'shape_cache') and self.cache_enabled:
                previous_layer_id = id(self.previous)
                self.shape_cache[previous_layer_id] = previous_size
            return previous_size
        elif hasattr(self, '_input_shape'):
            return self._input_shape
        else:
            raise Exception('Layer is not connected. Did you forget to set "input_shape"?')

    def set_input_shape(self, input_shape):
        if type(input_shape) not in [tuple, list]:
            raise Exception('Invalid input shape - input_shape should be a tuple of int.')
        input_shape = tuple(input_shape)
        if hasattr(self, 'input_ndim') and self.input_ndim:
            if self.input_ndim != len(input_shape):
                raise Exception('Invalid input shape - Layer expects input ndim=' +
                                str(self.input_ndim) +
                                ', was provided with input shape ' + str(input_shape))
        self._input_shape = input_shape
        self.input = tf.placeholder(tf.float32, shape=self._input_shape)
        self.build()

    @property
    def output_shape(self):
        # default assumption: tensor shape unchanged.
        return self.input_shape

    def get_output(self, train=False):
        return self.get_input(train)

    def get_input(self, train=False):
        if hasattr(self, 'previous'):
            # to avoid redundant computations,
            # layer outputs are cached when possible.
            if hasattr(self, 'layer_cache') and self.cache_enabled:
                previous_layer_id = '%s_%s' % (id(self.previous), train)
                if previous_layer_id in self.layer_cache:
                    return self.layer_cache[previous_layer_id]
            previous_output = self.previous.get_output(train=train)
            if hasattr(self, 'layer_cache') and self.cache_enabled:
                previous_layer_id = '%s_%s' % (id(self.previous), train)
                self.layer_cache[previous_layer_id] = previous_output
            return previous_output
        elif hasattr(self, 'input'):
            return self.input
        else:
            raise Exception('Layer is not connected' +
                            ' and is not an input layer.')

    def supports_masked_input(self):
        '''Whether or not this layer respects the output mask of its previous
        layer in its calculations.
        If you try to attach a layer that does *not* support masked_input to
        a layer that gives a non-None output_mask(), an error will be raised.
        '''
        return False

    def get_output_mask(self, train=None):
        '''For some models (such as RNNs) you want a way of being able to mark
        some output data-points as "masked",
        so they are not used in future calculations.
        In such a model, get_output_mask() should return a mask
        of one less dimension than get_output()
        (so if get_output is (nb_samples, nb_timesteps, nb_dimensions),
        then the mask is (nb_samples, nb_timesteps),
        with a one for every unmasked datapoint,
        and a zero for every masked one.
        If there is *no* masking then it shall return None.
        For instance if you attach an Activation layer (they support masking)
        to a layer with an output_mask, then that Activation shall
        also have an output_mask.
        If you attach it to a layer with no such mask,
        then the Activation's get_output_mask shall return None.
        Some layers have an output_mask even if their input is unmasked,
        notably Embedding which can turn the entry "0" into
        a mask.
        '''
        return None

    def set_weights(self, weights):
        '''Set the weights of the layer.
        weights: a list of numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the layer (i.e. it should match the
            output of `get_weights`).
        '''
        params = self.trainable_weights + self.non_trainable_weights
        assert len(params) == len(weights), ('Provided weight array does not match layer weights (' +
                                             str(len(params)) + ' layer params vs. ' +
                                             str(len(weights)) + ' provided weights)')
        for p, w in zip(params, weights):
            if p.eval(session=_get_session()).shape != w.shape:
                raise Exception('Layer weight shape %s not compatible with provided weight shape %s.' %
                                (p.eval(session=_get_session()).shape, w.shape))
            tf.assign(p, np.asarray(w)).op.run(session=_get_session())

    def get_weights(self):
        '''Return the weights of the layer,
        as a list of numpy arrays.
        '''
        params = self.trainable_weights + self.non_trainable_weights
        weights = []
        for p in params:
            weights.append(p.eval(session=_get_session()))
        return weights

    def get_config(self):
        '''Return the parameters of the layer, as a dictionary.
        '''
        config = {'name': self.__class__.__name__}
        if hasattr(self, '_input_shape'):
            input_shape = self._input_shape
            if input_shape[0]:
                config['batch_input_shape'] = input_shape[:]
            else:
                config['input_shape'] = input_shape[1:]
        if hasattr(self, '_trainable'):
            config['trainable'] = self._trainable
        config['cache_enabled'] = self.cache_enabled
        config['custom_name'] = self.name
        return config

    def get_params(self):
        consts = []
        updates = []

        if hasattr(self, 'regularizers'):
            regularizers = self.regularizers
        else:
            regularizers = []

        if hasattr(self, 'constraints') and len(self.constraints) == len(self.trainable_weights):
            for c in self.constraints:
                if c:
                    consts.append(c)
                else:
                    consts.append(constraints.identity())
        elif hasattr(self, 'constraint') and self.constraint:
            consts += [self.constraint for _ in range(len(self.trainable_weights))]
        else:
            consts += [constraints.identity() for _ in range(len(self.trainable_weights))]

        if hasattr(self, 'updates') and self.updates:
            updates += self.updates

        return self.trainable_weights, regularizers, consts, updates

    def count_params(self):
        '''Return the total number of floats (or ints)
        composing the weights of the layer.
        '''
        shape = p.get_shape()
        params_count = np.prod([shape[i]._value for i in range(len(shape))])
        return sum([params_count for p in self.trainable_weights])


class MaskedLayer(Layer):
    '''If your layer trivially supports masking
    (by simply copying the input mask to the output),
    then subclass MaskedLayer instead of Layer,
    and make sure that you incorporate the input mask
    into your calculation of get_output().
    '''
    def supports_masked_input(self):
        return True

    def get_input_mask(self, train=False):
        if hasattr(self, 'previous'):
            return self.previous.get_output_mask(train)
        else:
            return None

    def get_output_mask(self, train=False):
        ''' The default output mask is just the input mask unchanged.
        Override this in your own implementations if,
        for instance, you are reshaping the input'''
        return self.get_input_mask(train)
    
    
class RecurrentConv2D(MaskedLayer):
    '''Abstract base class for recurrent layers.
    Do not use in a model -- it's not a functional layer!
    All recurrent layers (GRU, LSTM, SimpleRNN) also
    follow the specifications of this class and accept
    the keyword arguments listed below.
    # Input shape
        5D tensor with shape `(nb_samples, timesteps, channels,rows,cols)`.
    # Output shape
        - if `return_sequences`: 5D tensor with shape
            `(nb_samples, timesteps, channels,rows,cols)`.
        - else, 2D tensor with shape `(nb_samples, channels,rows,cols)`.
    # Arguments
        weights: list of numpy arrays to set as initial weights.
            The list should have 3 elements, of shapes:
            `[(input_dim, nb_filter), (nb_filter, nb_filter), (nb_filter,)]`.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, rocess the input sequence backwards.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
            is required when using this layer as the first layer in a model.
        input_shape: input_shape
    # Masking
        This layer supports masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an [Embedding](embeddings.md) layer with the `mask_zero` parameter
        set to `True`.
        **Note:** for the time being, masking is only supported with Theano.
    # TensorFlow warning
        For the time being, when using the TensorFlow backend,
        the number of timesteps used must be specified in your model.
        Make sure to pass an `input_length` int argument to your
        recurrent layer (if it comes first in your model),
        or to pass a complete `input_shape` argument to the first layer
        in your model otherwise.
    # Note on using statefulness in RNNs
        You can set RNN layers to be 'stateful', which means that the states
        computed for the samples in one batch will be reused as initial states
        for the samples in the next batch.
        This assumes a one-to-one mapping between
        samples in different successive batches.
        To enable statefulness:
            - specify `stateful=True` in the layer constructor.
            - specify a fixed batch size for your model, by passing
                a `batch_input_size=(...)` to the first layer in your model.
                This is the expected shape of your inputs *including the batch
                size*.
                It should be a tuple of integers, e.g. `(32, 10, 100)`.
        To reset the states of your model, call `.reset_states()` on either
        a specific layer, or on your entire model.
    '''
    input_ndim = 5

    def __init__(self, weights=None,
                 return_sequences=False, go_backwards=False, stateful=False,
                 nb_row=None, nb_col=None, nb_filter=None,
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful

        self.nb_row = nb_row
        self.nb_col = nb_col
        self.nb_filter = nb_filter

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        super(RecurrentConv2D, self).__init__(**kwargs)

    def get_output_mask(self, train=False):
        if self.return_sequences:
            return super(RecurrentConv2D, self).get_output_mask(train)
        else:
            return None

    @property
    def output_shape(self):

        input_shape = self.input_shape
        rows = input_shape[1+1]
        cols = input_shape[2+1]

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.return_sequences:
            return (input_shape[0], input_shape[1],
                    rows, cols, self.nb_filter)
        else:
            return (input_shape[0], rows, cols, self.nb_filter)

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, X, train=False):
        return None

    def get_initial_states(self, X):
        # (samples, timesteps, row, col, filter)
        initial_state = tf.zeros_like(X)
        # (samples,row, col, filter)
        axis = normalize_axis(axis, ndim(initial_state))
        initial_state =  tf.reduce_sum(initial_state, reduction_indices=1, keep_dims=False)
        # initial_state = initial_state[::,]
        zeros = np.zeros(self.W_shape, tf.float32)
        v = tf.Variable(np.asarray(zeros, dtype=tf.float32))
        _get_session().run(v.initializer)
        initial_state = self.conv_step(initial_state, v,
                                       border_mode=self.border_mode)

        initial_states = [initial_state for _ in range(2)]
        return initial_states

    def get_output(self, train=False):

        X = self.get_input(train)
        mask = self.get_input_mask(train)
        constants = self.get_constants(X, train)

        assert len(X.get_shape()) == 5
            
        if not self.input_shape[1]:
            raise Exception('When using TensorFlow, you should define ' +
                            'explicitely the number of timesteps of ' +
                            'your sequences. Make sure the first layer ' +
                            'has a "batch_input_shape" argument ' +
                            'including the samples axis.')

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        last_output, outputs, states = rnn(self.step, X,
                                           initial_states,
                                           go_backwards=self.go_backwards,
                                           mask=mask,
                                           constants=constants)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "return_sequences": self.return_sequences,
                  "go_backwards": self.go_backwards,
                  "stateful": self.stateful}
        if self.stateful:
            config['batch_input_shape'] = self.input_shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(RecurrentConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LSTMConv2D(RecurrentConv2D):
    '''
    # Input shape
            5D tensor with shape:
            `(samples,time, rows, cols, channels)`.
     # Output shape
        if return_sequences=False
            4D tensor with shape:
            `(samples, o_row, o_col, nb_filter)`.
        if return_sequences=True
            5D tensor with shape:
            `(samples, time, o_row, o_col, nb_filter)`.
        where o_row and o_col depend on the shape of the filter and
        the border_mode
        # Arguments
            nb_filter: Number of convolution filters to use.
            nb_row: Number of rows in the convolution kernel.
            nb_col: Number of columns in the convolution kernel.
            border_mode: 'valid' or 'same'.
            sub_sample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
            stateful : has not been checked yet.
            init: weight initialization function.
                Can be the name of an existing function (str),
                or a Theano function
                (see: [initializations](../initializations.md)).
            inner_init: initialization function of the inner cells.
            forget_bias_init: initialization function for the bias of the
            forget gate.
                [Jozefowicz et al.]
                (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
                recommend initializing with ones.
            activation: activation function.
                Can be the name of an existing function (str),
                or a Theano function (see: [activations](../activations.md)).
            inner_activation: activation function for the inner cells.
    # References
        - [Convolutional LSTM Network: A Machine Learning Approach for
        Precipitation Nowcasting](http://arxiv.org/pdf/1506.04214v1.pdf)
        The current implementation does not include the feedback loop on the
        cells output
    '''
    def __init__(self, nb_filter, nb_row, nb_col,
                 border_mode="SAME", sub_sample=(1, 1),
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = tt.initializations.glorot_uniform
        self.inner_init = tt.initializations.orthogonal
        self.forget_bias_init = tt.initializations.ones
        self.activation = tf.nn.tanh
        self.inner_activation = tt.network.hard_sigmoid
        self.border_mode = border_mode
        self.subsample = sub_sample

        kwargs["nb_filter"] = nb_filter
        kwargs["nb_row"] = nb_row
        kwargs["nb_col"] = nb_col

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.b_regularizer = b_regularizer
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        super(LSTMConv2D, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        self.input = tf.placeholder(tf.float32, shape=input_shape)

        stack_size = self.input_shape[3+1]
        self.W_shape = (self.nb_row, self.nb_col,
                        stack_size, self.nb_filter)

        self.W_shape1 = (self.nb_row, self.nb_col,
                         self.nb_filter, self.nb_filter)

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensor of shape (nb_filter)
            self.states = [None, None, None, None]

        self.W_i = self.init(self.W_shape)
        self.U_i = self.inner_init(self.W_shape1)
        zeros_i = np.zeros((self.nb_filter,))
        var_i = tf.Variable(np.asarray(zeros_i, dtype=tf.float32))
        _get_session().run(var_i.initializer)
        self.b_i = var_i

        self.W_f = self.init(self.W_shape)
        self.U_f = self.inner_init(self.W_shape1)
        self.b_f = self.forget_bias_init((self.nb_filter,))

        self.W_c = self.init(self.W_shape)
        self.U_c = self.inner_init(self.W_shape1)
        zeros_c = np.zeros((self.nb_filter,))
        var_c = tf.Variable(np.asarray(zeros_c, dtype=tf.float32))
        _get_session().run(var_c.initializer)
        self.b_c = var_c

        self.W_o = self.init(self.W_shape)
        self.U_o = self.inner_init(self.W_shape1)
        zeros_o = np.zeros((self.nb_filter,))
        var_o = tf.Variable(np.asarray(zeros_o, dtype=tf.float32))
        _get_session().run(var_o.initializer)
        self.b_o = var_o

        def append_regulariser(input_regulariser, param, regularizers_list):
            regulariser = regularizers.get(input_regulariser)
            if regulariser:
                regulariser.set_param(param)
                regularizers_list.append(regulariser)

        self.regularizers = []
        for W in [self.W_i, self.W_f, self.W_i, self.W_o]:
            append_regulariser(self.W_regularizer, W, self.regularizers)
        for U in [self.U_i, self.U_f, self.U_i, self.U_o]:
            append_regulariser(self.U_regularizer, U, self.regularizers)
        for b in [self.b_i, self.b_f, self.b_i, self.b_o]:
            append_regulariser(self.b_regularizer, b, self.regularizers)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided ' +
                            '(including batch size).')

        if self.return_sequences:
            out_row, out_col, out_filter = self.output_shape[2:]
        else:
            out_row, out_col, out_filter = self.output_shape[1:]

        if hasattr(self, 'states'):
            tf.assign(self.states[0],
                      np.asarray(np.zeros((input_shape[0], out_row, out_col, out_filter)))).op.run(session=_get_session())
            tf.assign(self.states[1],
                      np.asarray(np.zeros((input_shape[0], out_row, out_col, out_filter)))).op.run(session=_get_session())
        else:
            zeros1 = np.zeros((input_shape[0], out_row, out_col, out_filter))
            state1 = tf.Variable(np.asarray(zeros1, dtype=tf.float32))
            _get_session().run(state1.initializer)
            zeros2 = np.zeros((input_shape[0], out_row, out_col, out_filter))
            state2 = tf.Variable(np.asarray(zeros2, dtype=tf.float32))
            _get_session().run(state2.initializer)
            self.states = [state1, state2]

    def conv_step(self, x, W, b=None, border_mode="SAME"):

        conv_out = conv2d(x, W, strides=self.subsample,
                          border_mode=border_mode,
                          image_shape=(self.input_shape[0],
                                       self.input_shape[2],
                                       self.input_shape[3],
                                       self.input_shape[4]),
                          filter_shape=self.W_shape)
        if b:
            conv_out = conv_out + tf.reshape(b, (1, 1, 1, self.nb_filter))


        return conv_out

    def conv_step_hidden(self, x, W, border_mode="SAME"):
        # This new function was defined because the
        # image shape must be hardcoded

        if self.return_sequences:
            out_row, out_col, out_filter = self.output_shape[2:]
        else:
            out_row, out_col, out_filter = self.output_shape[1:]

        conv_out = conv2d(x, W, strides=(1, 1),
                          border_mode=border_mode,
                          image_shape=(self.input_shape[0],
                                       out_row, out_col,
                                       out_filter),
                          filter_shape=self.W_shape1)

        return conv_out

    def step(self, x, states):
        assert len(states) == 4
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_W = states[2]
        B_U = states[3]

        x_i = self.conv_step(x * B_W[0], self.W_i, self.b_i,
                             border_mode=self.border_mode)
        x_f = self.conv_step(x * B_W[1], self.W_f, self.b_f,
                             border_mode=self.border_mode)
        x_c = self.conv_step(x * B_W[2], self.W_c, self.b_c,
                             border_mode=self.border_mode)
        x_o = self.conv_step(x * B_W[3], self.W_o, self.b_o,
                             border_mode=self.border_mode)

        # U : from nb_filter to nb_filter
        # Same because must be stable in the ouptut space
        h_i = self.conv_step_hidden(h_tm1, self.U_i * B_U[0],
                                    border_mode="same")
        h_f = self.conv_step_hidden(h_tm1, self.U_f * B_U[1],
                                    border_mode="same")
        h_c = self.conv_step_hidden(h_tm1, self.U_c * B_U[2],
                                    border_mode="same")
        h_o = self.conv_step_hidden(h_tm1, self.U_o * B_U[3],
                                    border_mode="same")

        i = self.inner_activation(x_i + h_i)
        f = self.inner_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.inner_activation(x_o + h_o)
        h = o * self.activation(c)

        return h, [h, c]

    def get_constants(self, X, train=False):
        retain_p_W = 1. - self.dropout_W
        retain_p_U = 1. - self.dropout_U
        if train and (self.dropout_W > 0 or self.dropout_U > 0):
            nb_samples = X.get_shape()[0]
            
            if not self.input_shape[0]:
                raise Exception('For RNN dropout in tensorflow, ' +
                                'a complete input_shape must be ' +
                                'provided (including batch size).')
            nb_samples = self.input_shape[0]
            
            in_shape = (nb_samples, self.input_dim)
            random_bin_in = tf.select(
                tf.random_uniform(in_shape, dtype=tf.float32, seed=np.random.randint(10e6)) <= retain_p_W, 
                tf.ones(in_shape), tf.zeros(in_shape))
            B_W = [random_bin_in for _ in range(4)]
            
            out_shape = (nb_samples, self.output_dim)
            random_bin_out = tf.select(
                tf.random_uniform(out_shape, dtype=tf.float32, seed=np.random.randint(10e6)) <= retain_p_U, 
                tf.ones(out_shape), tf.zeros(out_shape))
            B_U = [random_bin_out for _ in range(4)]
        else:
            B_W = np.ones(4, dtype=np.float32) * retain_p_W
            B_U = np.ones(4, dtype=np.float32) * retain_p_U
        return [B_W, B_U]

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "nb_filter": self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  'border_mode': self.border_mode,
                  "inner_activation": self.inner_activation.__name__}
        base_config = super(LSTMConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""