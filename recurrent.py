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

from tensorflow.python.ops.nn_ops import conv2d

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


def rnn_conv2d(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
    """Creates a recurrent neural network specified by RNNConv2DCell `cell`.
    The simplest form of RNN network generated is:
        state = cell.zero_state(...)
        outputs = []
        for input_ in inputs:
            output, state = cell(input_, state)
            outputs.append(output)
        return (outputs, state)
        
        However, a few other options are available:
        An initial state can be provided.
        If the sequence_length vector is provided, dynamic calculation is performed.
        This method of calculation does not compute the RNN steps past the maximum
        sequence length of the minibatch (thus saving computational time),
        and properly propagates the state at an example's sequence length
        to the final state output.
        The dynamic calculation performed is, at time t for batch row b,
            (output, state)(b, t) =
                (t >= sequence_length(b))
                    ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
                    : cell(input(b, t), state(b, t - 1))
    Args:
        cell: An instance of RNNCell.
        inputs: A length T list of inputs, each a tensor of shape
            [batch_size, input_size].
        initial_state: (optional) An initial state for the RNN.
            If `cell.state_size` is an integer, this must be
            a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
            If `cell.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state.  Required if
            initial_state is not provided.
        sequence_length: Specifies the length of each sequence in inputs.
                         An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
        scope: VariableScope for the created subgraph; defaults to "RNN".
    Returns:
        A pair (outputs, state) where:
        - outputs is a length T list of outputs (one for each input)
        - state is the final state
    Raises:
        TypeError: If `cell` is not an instance of RNNConv2DCell.
        ValueError: If `inputs` is `None` or an empty list, or if the input depth
                    (column size) cannot be inferred from inputs via shape inference.
    """

    if not isinstance(cell, RNNConv2DCell):
        raise TypeError("cell must be an instance of RNNConv2DCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "RNNConv2D") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        if inputs[0].get_shape().ndims != 1:
            (fixed_batch_size, input_height, input_width, input_channels) = inputs[0].get_shape().with_rank(4)
            if input_height.value is None or input_width.value is None or input_channels.value is None:
                raise ValueError(
                    "Input size (2nd, 3rd or 4th dimension of inputs[0]) must be accessible via "
                    "shape inference, but saw value None.")
        else:
            fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(inputs[0])[0]
        if initial_state is not None:
            state = initial_state
            print("state not None c: ", state[0])
            print("state not None h: ", state[1])
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)
            print("state None c: ", state[0])
            print("state None h: ", state[1])

        if sequence_length is not None:
            # Prepare variables
            sequence_length = math_ops.to_int32(sequence_length)
            zero_output = array_ops.zeros(
                array_ops.pack([batch_size,
                                cell.output_size[0],
                                cell.output_size[1],
                                cell.output_size[2]]),
                               inputs[0].dtype)
            zero_output.set_shape(
                tensor_shape.TensorShape([fixed_batch_size.value,
                                          cell.output_size[0],
                                          cell.output_size[1],
                                          cell.output_size[2]]))
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = tf.nn.rnn._rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (output, state) = call_cell()

            outputs.append(output)

        return (outputs, state)
    
    
def _packed_state_with_indices(structure, flat, index):
    """Helper function for _packed_state.
    Args:
        structure: Substructure (tuple of elements and/or tuples) to mimic
        flat: Flattened values to output substructure for.
        index: Index at which to start reading from flat.
    Returns:
        The tuple (new_index, child), where:
        * new_index - the updated index into `flat` having processed `structure`.
        * packed - the subset of `flat` corresponding to `structure`,
                   having started at `index`, and packed into the same nested
                   format.
    Raises:
        ValueError: if `structure` contains more elements than `flat`
        (assuming indexing starts from `index`).
    """
    packed = []
    print('structure', structure)
    print('flat', flat)
    for s in structure:
        print('s', s)
        print('index', index)
        if tf.nn.rnn_cell._is_sequence(s):
            print('pswi: is sequence')
            new_index, child = _packed_state_with_indices(s, flat, index)
            print('new_index', new_index)
            print('child', child)
            packed.append(tf.nn.rnn_cell._sequence_like(s, child))
            index = new_index
        else:
            print('pswi: is no sequence')
            
            packed.append(flat[index])
            index += 1
    return (index, packed)


def _packed_state(structure, state):
    """Returns the flat state packed into a recursive tuple like structure.
    Args:
        structure: tuple or list constructed of scalars and/or other tuples/lists.
        state: flattened state.
    Returns:
        packed: `state` converted to have the same recursive structure as
                `structure`.
    Raises:
        TypeError: If structure or state is not a tuple or list.
        ValueError: If state and structure have different element counts.
    """
    if not tf.nn.rnn_cell._is_sequence(structure):
        raise TypeError("structure must be a sequence")
    if not tf.nn.rnn_cell._is_sequence(state):
        raise TypeError("state must be a sequence")
    
    print('structure: ', structure)
    # flat_structure = tf.nn.rnn_cell._unpacked_state(structure)
    flat_structure = (structure[0], structure[1])
    print('flat structure: ', flat_structure)
    print('size: flat: {}, state: {}'.format(len(flat_structure), len(state)))
    if len(flat_structure) != len(state):
        raise ValueError(
            "Internal error: Could not pack state.  Structure had %d elements, but "
            "state had %d elements.  Structure: %s, state: %s."
            % (len(flat_structure), len(state), structure, state))

    #(_, packed) = _packed_state_with_indices(structure, state, 0)
    packed = [state[0], state[1]]
    return tf.nn.rnn_cell._sequence_like(structure, packed)
    
    
class RNNConv2DCell(object):
    """Abstract object representing an 2D convolutional RNN cell.
    An Conv2D-RNN cell, in the most abstract setting, is anything that has
    a state and performs some operation that takes a matrix of inputs.
    This operation results in an output matrix with `self.output_size` columns.
    If `self.state_size` is an integer, this operation also results in a new
    state matrix with `self.state_size` columns.  If `self.state_size` is a
    tuple of integers, then it results in a tuple of `len(state_size)` state
    matrices, each with the a column size corresponding to values in `state_size`.
    This module provides a number of basic commonly used RNN cells, such as
    LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
    of operators that allow add dropouts, projections, or embeddings for inputs.
    Constructing multi-layer cells is supported by the class `MultiRNNCell`,
    or by calling the `rnn` ops several times. Every `RNNConv2DCell` must have the
    properties below and and implement `__call__` with the following signature.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNNConv2D cell on inputs, starting from the given state.
        Args:
            inputs: `4-D` tensor with shape `[batch_size x height x width x filters]`.
            state: if `self.state_size` is an integer, this should be a `2-D Tensor`
                   with shape `[batch_size x height x width x filters]`.  Otherwise, if
                   `self.state_size` is a tuple of integers, this should be a tuple
                   with shapes `[batch_size x s] for s in self.state_size`.
            scope: VariableScope for the created subgraph; defaults to class name.
        Returns:
            A pair containing:
            - Output: A `4-D` tensor with shape `[batch_size x height x width x filters]`.
            - New state: Either a single `4-D` tensor, or a tuple of tensors matching
              the arity and shapes of `state`.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.
        Returns:
            If `state_size` is an int, then the return value is a `2-D` tensor of
            shape `[batch_size x state_size]` filled with zeros.
            If `state_size` is a nested list or tuple, then the return value is
            a nested list or tuple (of the same structure) of `2-D` tensors with
            the shapes `[batch_size x s]` for each s in `state_size`.
        """
        state_size = self.state_size
        print('state-size: ', state_size)
        if tf.nn.rnn_cell._is_sequence(state_size):
            #state_size_flat = tf.nn.rnn_cell._unpacked_state(state_size)
            state_size_flat = (state_size.c, state_size.h)
            print('state_size_flat', state_size_flat)
            zeros_flat = [
                array_ops.zeros(array_ops.pack([batch_size, s[0], s[1], s[2]]), dtype=dtype)
                for s in state_size_flat]
            for s, z in zip(state_size_flat, zeros_flat):
                z.set_shape([None, s[0], s[1], s[2]])
            zeros = _packed_state(structure=state_size, state=zeros_flat)
        else:
            raise ValueError('Internal state is not a sequence.')
        print('zeros', zeros)
        return zeros



class BasicLSTMConv2DCell(RNNConv2DCell):
    """Basic 2D convolutional LSTM recurrent network cell.
    The implementation is based on: https://arxiv.org/abs/1506.04214.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMConv2DCell that follows.
    """

    def __init__(self, nb_rows, nb_cols, nb_filters, height, width, forget_bias=1.0, activation=tanh):
        """Initialize the basic 2D convolutional LSTM cell.
        Parameters
        ----------
        nb_rows: int 
            The number of rows of the convolutioanl kernel.
        nb_cols: int 
            The number of columns of the convolutioanl kernel.
        nb_filters: int
            The number of filters of the convolutional kernel. This also specifies
            the depth/channels of the output.
        height: int
            The height of the input image.
        width: int
            The width of the input image.
        forget_bias: float
            The bias added to forget gates (see above).
        activation: function
            Activation function of the inner states.
        """
        self._nb_rows = nb_rows
        self._nb_cols = nb_cols
        self._nb_filters = nb_filters
        self._height = height
        self._width = width
        self._forget_bias = forget_bias
        self._state_is_tuple = True
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(
            (self._height, self._width, self._nb_filters), 
            (self._height, self._width, self._nb_filters))

    @property
    def output_size(self):
        return (self._height, self._width, self._nb_filters)

    def __call__(self, inputs, state, scope=None):
        """2D convolutional Long short-term memory cell (LSTMConv2D)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMConv2DCell"
            c, h = state
            
            print('c', c)
            print('h', h)
                
            concat = _conv2d(inputs, h, self._nb_rows, self._nb_cols, self._nb_filters , True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i = concat[0]
            j = concat[1]
            f = concat[2]
            o = concat[3]
            
            print('i', i.get_shape())
            print('j', j.get_shape())
            print('f', f.get_shape())
            print('o', o.get_shape())

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                 self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)
            
            # i_t = sig(i)
            # f_t = sig(f + b_f)
            # new_c = f_t * c + i_t * tanh(j)
            # o_t = sig(o)
            # new_h = o_t * tanh(new_c)

            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

        
def _conv2d(inputs, hidden, rows, cols, filters, bias, bias_start=0.0, scope=None):
    """Conv map: sum_i(conv(args[i], W[i])), where W[i] is a variable.
    Args:
        args: a 4D Tensor or a list of 4D, batch x height x width x filters, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 4D Tensor with shape [batch x height x width x filters] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Now the computation.
    with vs.variable_scope(scope or "Conv"):
        input_channels = inputs.get_shape()[-1]

        k_inputs1 = vs.get_variable("Kernel_w1", [rows, cols, input_channels, filters]) # TODO: use channels_in from input
        k_inputs2 = vs.get_variable("Kernel_w2", [rows, cols, input_channels, filters]) # TODO: use channels_in from input
        k_inputs3 = vs.get_variable("Kernel_w3", [rows, cols, input_channels, filters]) # TODO: use channels_in from input
        k_inputs4 = vs.get_variable("Kernel_w4", [rows, cols, input_channels, filters]) # TODO: use channels_in from input
        
        k_hidden1 = vs.get_variable("Kernel_u1", [rows, cols, filters, filters]) # TODO: use channels_in from input
        k_hidden2 = vs.get_variable("Kernel_u2", [rows, cols, filters, filters]) # TODO: use channels_in from input
        k_hidden3 = vs.get_variable("Kernel_u3", [rows, cols, filters, filters]) # TODO: use channels_in from input
        k_hidden4 = vs.get_variable("Kernel_u4", [rows, cols, filters, filters]) # TODO: use channels_in from input

        conv_inputs1 = conv2d(inputs, k_inputs1, [1, 1, 1, 1], "SAME")
        conv_inputs2 = conv2d(inputs, k_inputs2, [1, 1, 1, 1], "SAME")
        conv_inputs3 = conv2d(inputs, k_inputs3, [1, 1, 1, 1], "SAME")
        conv_inputs4 = conv2d(inputs, k_inputs4, [1, 1, 1, 1], "SAME")
        
        conv_hidden1 = conv2d(hidden, k_hidden1, [1, 1, 1, 1], "SAME")
        conv_hidden2 = conv2d(hidden, k_hidden2, [1, 1, 1, 1], "SAME")
        conv_hidden3 = conv2d(hidden, k_hidden3, [1, 1, 1, 1], "SAME")
        conv_hidden4 = conv2d(hidden, k_hidden4, [1, 1, 1, 1], "SAME")
        
        res1 = conv_inputs1 + conv_hidden1
        res2 = conv_inputs2 + conv_hidden2
        res3 = conv_inputs3 + conv_hidden3
        res4 = conv_inputs4 + conv_hidden4
        
        if not bias:
            return [res1, res2, res3, res4]
        bias_term1 = vs.get_variable(
            "Bias1", [filters],
            initializer=init_ops.constant_initializer(bias_start))
        bias_term2 = vs.get_variable(
            "Bias2", [filters],
            initializer=init_ops.constant_initializer(bias_start))
        bias_term3 = vs.get_variable(
            "Bias3", [filters],
            initializer=init_ops.constant_initializer(bias_start))
        bias_term4 = vs.get_variable(
            "Bias4", [filters],
            initializer=init_ops.constant_initializer(bias_start))
    return [res1 + bias_term1, res2 + bias_term2, res3 + bias_term3, res4 + bias_term4]
