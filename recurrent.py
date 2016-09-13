from __future__ import absolute_import
from __future__ import division

import collections
import six

import numpy as np
import tensorflow as tf
import tensortools as tt

from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.util import nest


def rnn_conv2d(cell, inputs, initial_state=None, dtype=tf.float32,
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
        scope: VariableScope for the created subgraph; defaults to "RNNConv2D".
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
            batch_size = tf.shape(inputs[0])[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:
            # Prepare variables
            sequence_length = tf.to_int32(sequence_length)
            zero_output = tf.zeros(
                tf.pack([batch_size,
                         cell.output_size[0],
                         cell.output_size[1],
                         cell.output_size[2]]),
                inputs[0].dtype)
            zero_output.set_shape(
                tensor_shape.TensorShape([fixed_batch_size.value,
                                          cell.output_size[0],
                                          cell.output_size[1],
                                          cell.output_size[2]]))
            min_sequence_length = tf.reduce_min(sequence_length)
            max_sequence_length = tf.reduce_max(sequence_length)

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
    
    
def rnn_conv2d_roundabout(cell, single_input, sequence_length, initial_state=None,
                          output_postprocessor=lambda x: x, dtype=tf.float32, scope=None):
    """Creates a recurrent neural network specified by RNNConv2DCell `cell`, that has only
       a single input, and reuses the last output as its new input.
    Args:
        cell: An instance of RNNCell.
        single_input: A singe input tensor of shape [batch_size, input_size].
        sequence_length: Specifies the length of the RNN.
                         This parameter is NOT the same as in the rnn_conv2d() or rnn() function!!!
        initial_state: (optional) An initial state for the RNN.
            If `cell.state_size` is an integer, this must be
            a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
            If `cell.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell.state_size`.
        output_postprocessor: function, optional
            A postprecessor function for the generated output each step.
        dtype: (optional) The data type for the initial state.  Required if
            initial_state is not provided.
        scope: VariableScope for the created subgraph; defaults to "RNNConv2D".
    Returns:
        A pair (outputs, state) where:
        - outputs is a length T list of outputs (one for each input)
        - state is the final state
    Raises:
        TypeError: If `cell` is not an instance of RNNConv2DCell.
        ValueError: If `single_input` is `None` or a list.
    """

    if not isinstance(cell, RNNConv2DCell):
        raise TypeError("cell must be an instance of RNNConv2DCell")
    if isinstance(single_input, list):
        raise TypeError("single_input must be no list")
    if single_input is None:
        raise ValueError("single_input must not be empty")

    outputs = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "RNNConv2D") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        if single_input.get_shape().ndims != 1:
            (fixed_batch_size, input_height, input_width, input_channels) = single_input.get_shape().with_rank(4)
            if input_height.value is None or input_width.value is None or input_channels.value is None:
                raise ValueError(
                    "Input size (2nd, 3rd or 4th dimension of single_input) must be accessible via "
                    "shape inference, but saw value None.")
        else:
            fixed_batch_size = single_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = tf.shape(single_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        input_ = single_input
        for time in xrange(sequence_length):
            if time > 0:
                varscope.reuse_variables()
            
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            (output, state) = call_cell()

            # use optional postprocessor
            output = output_postprocessor(output)
            
            outputs.append(output)
            input_ = output

        return (outputs, state)

    
def _sequence_like(instance, args):
    """Checks and returns sequence like structure.
    Taken from TensorFlow v0.9.
    """
    try:
        assert isinstance(instance, tuple)
        assert isinstance(instance._fields, collections.Sequence)
        assert all(isinstance(f, six.string_types) for f in instance._fields)
        # This is a namedtuple
        return type(instance)(*args)
    except (AssertionError, AttributeError):
        # Not a namedtuple
        return type(instance)(args)


    
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
        if nest.is_sequence(state_size):
            if isinstance(state_size, tf.nn.rnn_cell.LSTMStateTuple):
                # normal usage
                state_size_flat = (state_size.c, state_size.h)
                zeros_flat = [
                    tf.zeros(tf.pack([batch_size, s[0], s[1], s[2]]), dtype=dtype)
                    for s in state_size_flat]
                for s, z in zip(state_size_flat, zeros_flat):
                    z.set_shape([None, s[0], s[1], s[2]])
                zeros = _sequence_like(state_size, [zeros_flat[0], zeros_flat[1]])
            else:
                # when used with MultiRNNConv2DCell, it gets a tuple of state sizes
                layers = len(state_size)
                zeros_list = []
                for i in xrange(layers):
                    state_size_flat = (state_size[i].c, state_size[i].h)
                    zeros_flat = [
                        tf.zeros(tf.pack([batch_size, s[0], s[1], s[2]]), dtype=dtype)
                        for s in state_size_flat]
                    for s, z in zip(state_size_flat, zeros_flat):
                        z.set_shape([None, s[0], s[1], s[2]])
                    zeros_list.append(_sequence_like(state_size[0], [zeros_flat[0], zeros_flat[1]]))
                zeros = tuple(zeros_list)
        else:
            raise ValueError('Internal state is not a sequence.')
        return zeros


class BasicLSTMConv2DCell(RNNConv2DCell):
    """Basic 2D convolutional LSTM recurrent network cell.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMConv2DCell that follows.
    
    References:
    [Convolutional LSTM Network: A Machine Learning Approach for
    Precipitation Nowcasting](http://arxiv.org/pdf/1506.04214v1.pdf)
    The current implementation does not include the feedback loop on the
    cells output.
    """

    def __init__(self, height, width, n_filters, ksize_input, ksize_hidden,
                 weight_init=tf.contrib.layers.xavier_initializer(),
                 hidden_weight_init=tt.init.orthogonal_initializer(),
                 forget_bias=1.0,
                 activation=tf.nn.tanh, hidden_activation=tf.nn.sigmoid,
                 device=None):
        """Initialize the basic 2D convolutional LSTM cell.
        Parameters
        ----------
        height: int
            The height of the input image.
        width: int
            The width of the input image.
        n_filters: int
            The number of filters of the convolutional kernel. This also specifies
            the depth/channels of the output.
        ksize_hidden: tuple or list of (int, int) 
            The number of (rows, columns) of the convolutioanl kernel
            for input to state transition.
        ksize_hidden: tuple or list of (int, int) 
            The number of (rows, columns) of the convolutioanl kernel
            for state to state transition.
        weight_init : float or function, optional
            Initialization's of the input weights, either the standard deviation
            or a initializer-fuction such as xavier init.
        hidden_weight_init : float or function, optional
            Initialization's of the hidden weights, either the standard deviation
            or a initializer-fuction such as xavier or orthogonal init.
        forget_bias: float
            The bias added to forget gates (see above).
        activation: function
            Activation function of the output and cell states.
        hidden_activation: function
            Activation function of the hidden states.
        device: str or None, optional
            The device to which memory the variables will get stored on. (e.g. '/cpu:0')
        """
        self._height = height
        self._width = width
        self._n_filters = n_filters
        self._ksize_input = ksize_input
        self._ksize_hidden = ksize_hidden
        self._weight_init = weight_init
        self._hidden_weight_init = hidden_weight_init
        self._forget_bias = forget_bias
        self._state_is_tuple = True
        self._activation = activation
        self._hidden_activation = hidden_activation
        self._device = device

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(
            (self._height, self._width, self._n_filters), 
            (self._height, self._width, self._n_filters))

    @property
    def output_size(self):
        return (self._height, self._width, self._n_filters)

    def __call__(self, inputs, state, scope=None):
        """2D convolutional Long short-term memory cell (LSTMConv2D)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMConv2DCell"
            c, h = state
            with tf.variable_scope("Conv_x"):
                conv_xi = tt.network.conv2d("xi", inputs, self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=0.0,
                                            device=self._device)
                conv_xj = tt.network.conv2d("xj", inputs,self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=0.0,
                                            device=self._device)
                conv_xf = tt.network.conv2d("xf", inputs, self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=self._forget_bias,
                                            device=self._device)
                conv_xo = tt.network.conv2d("xo", inputs, self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=0.0,
                                            device=self._device)
            with tf.variable_scope("Conv_h"):
                conv_hi = tt.network.conv2d("hi", h, self._n_filters, 
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)
                conv_hj = tt.network.conv2d("hj", h, self._n_filters,
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)
                conv_hf = tt.network.conv2d("hf", h, self._n_filters,
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)
                conv_ho = tt.network.conv2d("ho", h, self._n_filters,
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)

            i = conv_xi + conv_hi  # input gate
            j = conv_xj + conv_hj  # new input
            f = conv_xf + conv_hf  # forget gate
            o = conv_xo + conv_ho  # output gate

            # i_t = sig(i)
            # f_t = sig(f + b_f)
            # new_c = f_t * c + i_t * tanh(j)
            # o_t = sig(o)
            # new_h = o_t * tanh(new_c)
            new_c = (c * self._hidden_activation(f) + self._hidden_activation(i) *
                 self._activation(j))
            new_h = self._activation(new_c) * self._hidden_activation(o)

            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

        
class LSTMConv2DCell(RNNConv2DCell):
    """2D convolutional LSTM recurrent network cell.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It allows cell clipping and supports peep-hole connections, as well as
    batch-normalization.
    
    As proposed in the RNN-BatchNorm paper, we do not center the input-hidden connections
    and rely on the learned forget-bias (center=False).
    Additionally, we optionally support BN for peephole-connections, which are not used
    in the paper. Use them with caution!!!
    
    References:
    [Convolutional LSTM Network: A Machine Learning Approach for
    Precipitation Nowcasting](http://arxiv.org/pdf/1506.04214)
    [Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025)
    """

    def __init__(self, height, width, n_filters, ksize_input, ksize_hidden,
                 use_peepholes=False, cell_clip=None,
                 bn_input_hidden=False, bn_hidden_hidden=False, bn_peepholes=False,
                 updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=None,
                 weight_init=tf.contrib.layers.xavier_initializer(),
                 hidden_weight_init=tt.init.orthogonal_initializer(),
                 forget_bias=1.0,
                 activation=tf.nn.tanh, hidden_activation=tf.nn.sigmoid,
                 device=None):
        """Initialize the basic 2D convolutional LSTM cell.
        Parameters
        ----------
        height: int
            The height of the input image.
        width: int
            The width of the input image.
        n_filters: int
            The number of filters of the convolutional kernel. This also specifies
            the depth/channels of the output.
        ksize_hidden: tuple or list of (int, int) 
            The number of (rows, columns) of the convolutioanl kernel
            for input to state transition. Usually ksize_hidden > ksize_input
        ksize_hidden: tuple or list of (int, int) 
            The number of (rows, columns) of the convolutioanl kernel
            for state to state transition. Usually ksize_hidden > ksize_input
        use_peepholes: Boolean, optioanl
            Set to True to enable diagonal/peephole connections.
        cell_clip: float, optional
            A float (> 0) value, if provided the cell state is clipped
            by this value prior to the cell output activation.
            Both sides are clipped (range [-cell_clip, cell_clip]).
        bn_input_hidden: Boolean, optional
            Set to true to enable batch-normalization for input-hidden connections,
            excluding peepholes. Requires to set 'is_training' param.
        bn_hidden_hidden: Boolean, optional
            Set to true to enable batch-normalization for hidden-hidden connections,
            excluding peepholes. Requires to set 'is_training' param.
        bn_peepholes: Boolean, optional
            Set to true to enable batch-normalization for peephole connections
            Requires to set 'is_training' param.
        updates_collections: str or None
            Collections to collect the update ops for computation in case of
            batch normalization is used. If None, a control dependency would be
            added to make sure the updates are computed. This is only required
            when any kind of batch_normalization is used, else ignored.
        is_training: Boolean or None, optional
            Whether or not the layer is in training mode. This is only required
            when any kind of batch_normalization is used, else ignored.
        weight_init : float or function, optional
            Initialization's of the input weights, either the standard deviation
            or a initializer-fuction such as xavier init.
        hidden_weight_init : float or function, optional
            Initialization's of the hidden weights, either the standard deviation
            or a initializer-fuction such as xavier or orthogonal init.
        forget_bias: float
            The bias added to forget gates (see above).
        activation: function
            Activation function of the output and cell states.
        hidden_activation: function
            Activation function of the hidden states.
        device: str or None, optional
            The device to which memory the variables will get stored on. (e.g. '/cpu:0')
        """
        if bn_input_hidden or bn_hidden_hidden or bn_peepholes:
            assert is_training is not None, "When BatchNorm is used," \
                                            "training mode has to be specified."
        
        if bn_peepholes:
            assert use_peepholes, "Enabling batch-norm for peepholes requires to enbale them."
        
        self._height = height
        self._width = width
        self._n_filters = n_filters
        self._ksize_input = ksize_input
        self._ksize_hidden = ksize_hidden
        self._weight_init = weight_init
        self._hidden_weight_init = hidden_weight_init
        self._forget_bias = forget_bias
        self._state_is_tuple = TrueIn training mode
        self._activation = activation
        self._hidden_activation = hidden_activation
        self._device = device
        
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._bn_input_hidden = bn_input_hidden
        self._bn_hidden_hidden = bn_hidden_hidden
        self._bn_peepholes = bn_peepholes
        self._updates_collections = updates_collections
        self._is_training = is_training

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(
            (self._height, self._width, self._n_filters), 
            (self._height, self._width, self._n_filters))

    @property
    def output_size(self):
        return (self._height, self._width, self._n_filters)

    def __call__(self, inputs, state, scope=None):
        """2D convolutional Long short-term memory cell (LSTMConv2D)."""
        with vs.variable_scope(scope or type(self).__name__):  # "LSTMConv2DCell"
            c, h = state
            with tf.variable_scope("Conv_x") as varscope:
                conv_xi = tt.network.conv2d("xi", inputs, self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=0.0,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_xi = tf.contrib.layers.batch_norm(conv_xi, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)
                    
                    # reuse the same batch-variables in the next 2d-convs
                    varscope.reuse_variables()
                
                conv_xj = tt.network.conv2d("xj", inputs,self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=0.0,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_xj = tf.contrib.layers.batch_norm(conv_xj, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)
                
                conv_xf = tt.network.conv2d("xf", inputs, self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=self._forget_bias,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_xf = tf.contrib.layers.batch_norm(conv_xf, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)
                
                conv_xo = tt.network.conv2d("xo", inputs, self._n_filters,
                                            self._ksize_input, (1, 1),
                                            weight_init=self._weight_init,
                                            bias_init=0.0,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_xo = tf.contrib.layers.batch_norm(conv_xo, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)

            with tf.variable_scope("Conv_h") as varscope:
                conv_hi = tt.network.conv2d("hi", h, self._n_filters, 
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_hi = tf.contrib.layers.batch_norm(conv_hi, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)
                    
                    # reuse the same batch-variables in the next 2d-convs
                    varscope.reuse_variables()
                
                conv_hj = tt.network.conv2d("hj", h, self._n_filters,
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_hj = tf.contrib.layers.batch_norm(conv_hj, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)
                
                conv_hf = tt.network.conv2d("hf", h, self._n_filters,
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_hf = tf.contrib.layers.batch_norm(conv_hf, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)
                
                conv_ho = tt.network.conv2d("ho", h, self._n_filters,
                                            self._ksize_hidden, (1, 1),
                                            weight_init=self._hidden_weight_init,
                                            bias_init=None,
                                            device=self._device)
                
                if self._bn_input_hidden:
                    conv_ho = tf.contrib.layers.batch_norm(conv_ho, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)

                i = conv_xi + conv_hi  # input gate
                j = conv_xj + conv_hj  # new input
                f = conv_xf + conv_hf  # forget gate
                o = conv_xo + conv_ho  # output gate
            
            if self._use_peepholes:
                # peepholes for input
                with tf.variable_scope("Conv_c"):
                    conv_ci = tt.network.conv2d("ci", c, self._n_filters, 
                                                self._ksize_hidden, (1, 1),
                                                weight_init=self._weight_init,
                                                bias_init=None,
                                                device=self._device)
                    
                    if self._bn_peepholes:
                        conv_ci = tf.contrib.layers.batch_norm(conv_ci, scale=True,
                        center=False, updates_collections=self._updates_collections,
                        is_trianing=self._is_training)
                    
                        # reuse the same batch-variables in the next 2d-convs
                        varscope.reuse_variables()
                    
                    conv_cf = tt.network.conv2d("cf", c, self._n_filters,
                                                self._ksize_hidden, (1, 1),
                                                weight_init=self._weight_init,
                                                bias_init=None,
                                                device=self._device)
                    
                    if self._bn_peepholes:
                        conv_cf = tf.contrib.layers.batch_norm(conv_cf, scale=True,
                            center=False, updates_collections=self._updates_collections,
                            is_trianing=self._is_training)
                    
                    i += conv_ci
                    f += conv_cf

            # i_t = sig(i)
            # f_t = sig(f + b_f)
            # new_c = f_t * c + i_t * tanh(j)
            # o_t = sig(o)
            # new_h = o_t * tanh(new_c)
            new_c = (c * self._hidden_activation(f) + self._hidden_activation(i) *
                 self._activation(j))
            
            if self._cell_clip is not None:
                # cell clip before output
                new_c = clip_ops.clip_by_value(new_c, -self._cell_clip, self._cell_clip)
            
            if self._use_peepholes:
                # peepholes for output
                with tf.variable_scope("Conv_c"):
                    conv_co = tt.network.conv2d("co", new_c, self._n_filters,
                                                self._ksize_hidden, (1, 1),
                                                weight_init=self._weight_init,
                                                bias_init=None,
                                                device=self._device)
                    
                    if self._bn_peepholes:
                        # reuse the same batch-variables from same scope above
                        varscope.reuse_variables()
                        
                        conv_co = tf.contrib.layers.batch_norm(conv_co, scale=True,
                            center=False, updates_collections=self._updates_collections,
                            is_trianing=self._is_training)
                    
                    o += conv_co
            
            if self._bn_hidden_hidden:
                new_c = tf.contrib.layers.batch_norm(new_c,
                                                     center=True, scale=True,
                                                     updates_collections=self._updates_collections,
                                                     is_trianing=self._is_training)
            
            new_h = self._activation(new_c) * self._hidden_activation(o)

            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state
        
        
class MultiRNNConv2DCell(RNNConv2DCell):
    """2D convolutional RNN cell composed sequentially of multiple simple cells."""

    def __init__(self, cells, state_is_tuple=True):
        """Create a RNN cell composed sequentially of a number of RNNConv2DCells.
        Args:
            cells: list of RNNConv2DCells that will be composed in this order.
            state_is_tuple: Just kept for compatibility, not used internally.
        Raises:
            ValueError: if cells is empty (not allowed), or at least one of the cells
                        returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNConv2DCell.")
        self._cells = cells
        self._state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNConv2DCell"
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with vs.variable_scope("Cell%d" % i):
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s"
                            % (len(self.state_size), state))
                    cur_state = state[i]
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)
        new_states = (tuple(new_states) if self._state_is_tuple
                      else tf.concat(1, new_states))
        return cur_inp, new_states
    

    
class BatchNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell): # TODO: remove this code!!!
    """Batch normalized LSTM cell.
    References:
        Code:  http://olavnymoen.com/2016/07/07/rnn-batch-normalization
        Paper: arxiv.org/abs/1603.09025"""
    def __init__(self, num_units, state_is_tuple=True,
                 device=None):
        self._num_units = num_units
        self._device = device

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tt.network.get_variable('W_xh',
                [x_size, 4 * self._num_units],
                initializer=tt.init.orthogonal_initializer(),
                device=self._device)
            W_hh = tt.network.get_variable('W_hh',
                [self._num_units, 4 * self._num_units],
                initializer=tt.init.bn_lstm_identity_initializer(0.95),
                device=self._device)
            bias = tt.network.get_variable('bias', [4 * self._num_units],
                                           device=self._device)

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            mean_xh, var_xh = tf.nn.moments(xh, [0])
            xh_scale = tt.network.get_variable('xh_scale', [4 * self._num_units], initializer=tf.constant_initializer(0.1),
                                               device=self._device)

            mean_hh, var_hh = tf.nn.moments(hh, [0])
            hh_scale = tt.network.get_variable('hh_scale', [4 * self._num_units], initializer=tf.constant_initializer(0.1),
                                               device=self._device)

            static_offset = tf.constant(0, dtype=tf.float32, shape=[4 * self._num_units])
            epsilon = 1e-3

            bn_xh = tf.nn.batch_normalization(xh, mean_xh, var_xh, static_offset, xh_scale, epsilon)
            bn_hh = tf.nn.batch_normalization(hh, mean_hh, var_hh, static_offset, hh_scale, epsilon)

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(1, 4, hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)

            mean_c, var_c = tf.nn.moments(new_c, [0])
            c_scale = tt.network.get_variable('c_scale', [self._num_units], initializer=tf.constant_initializer(0.1),
                                              device=self._device)
            c_offset = tt.network.get_variable('c_offset', [self._num_units],
                                               device=self._device)

            bn_new_c = tf.nn.batch_normalization(new_c, mean_c, var_c, c_offset, c_scale, epsilon)

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)

        
class BasicLSTMCell(tf.nn.rnn_cell.RNNCell):
    """Basic LSTM recurrent network cell.
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMCell that follows.
    
    This is just a copy of the original code in TensorFlow. The only modification
    is to use device-string for Multi-GPU environment. Remove this in case
    there is a way to assign variables to a device in a different way!!!
    """

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, activation=tf.tanh, device=None):
        """Initialize the basic LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (see above).
            input_size: Deprecated and unused.
            state_is_tuple: If True, accepted and returned states are 2-tuples of
                            the `c_state` and `m_state`.  If False, they are concatenated
                            along the column axis.  The latter behavior will soon be deprecated.
            activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            print("%s: Using a concatenated state is slower and will soon be "
                  "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            print("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._device = device

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
                c, h = tf.split(1, 2, state)
            concat = _linear([inputs, h], 4 * self._num_units, True, device=self._device)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])
            return new_h, new_state
            

def _linear(args, output_size, bias, bias_start=0.0, scope=None, device=None):
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
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
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

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = tt.network.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype,
            device=device)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tt.network.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype),
            device=device)
    return res + bias_term