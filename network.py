import types
import tensorflow as tf
from tensorflow.python.framework import ops


def get_variable(name, shape=None, dtype=tf.float32, initializer=None, regularizer=None,
                 trainable=True, collections=None, caching_device=None, partitioner=None, 
                 validate_shape=True, device=None):
    """Gets an existing variable with these parameters or create a new one.
       Extends TensorFlow's default function with the ability
       to assign it on a specific device
    Parameters
    ----------
    name: The name of the new or existing variable.
    shape: Shape of the new or existing variable.
    dtype: Type of the new or existing variable (defaults to DT_FLOAT).
    initializer: Initializer for the variable if one is created.
    regularizer: A (Tensor -> Tensor or None) function
        The result of applying it on a newly created variable will be added to
        the collection GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If True also add the variable to the graph collection
               GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
    collections: List of graph collections keys to add the Variable to.
                 Defaults to [GraphKeys.VARIABLES] (see tf.Variable).
    caching_device: Optional device string or function describing where the Variable
                    should be cached for reading. Defaults to the Variable's device.
                    If not None, caches on another device. Typical use is to cache on
                    the device where the Ops using the Variable reside, to deduplicate
                    copying through Switch and other conditional statements.
    partitioner: Optional callable that accepts a fully defined TensorShape and
                 dtype of the Variable to be created, and returns a list of partitions
                 for each axis (currently only one axis can be partitioned).
    validate_shape: If False, allows the variable to be initialized with a value of unknown shape.
                    If True, the default, the shape of initial_value must be known.
    device: str or None, optional
        The device to which memory the variables will get stored on. (e.g. '/cpu:0')
    Returns
    ----------
    The created or existing variable.

    Raises
    ----------
    ValueError: When creating a new variable and shape is not declared, 
                or when violating reuse during variable creation.
                Reuse is set inside variable_scope.
    """
    if device is not None:
        # specific device
        with tf.device(device):
            var = tf.get_variable(name, shape, dtype, initializer, regularizer,
                                  trainable, collections, caching_device, partitioner, 
                                  validate_shape)
    else:
        # default device
        var = tf.get_variable(name, shape, dtype, initializer, regularizer,
                              trainable, collections, caching_device, partitioner, 
                              validate_shape)
    return var


def lrelu(x, leak=0.2, name=None):
    """Leaky rectified linear unit.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.op_scope([x], name, 'lrelu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        x = tf.add(f1 * x, f2 * abs(x))
        return x


def hard_sigmoid(x, name=None):
    """Hard sigmoid implementation. This is a very rough approximation
       of a real sigmoid function, but is much faster to calculate.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    name : str, optional
        Variable scope to use.
    Returns
    ----------
    x: Tensor
        Output of the nonlinearity.
    """
    with tf.op_scope([x], name, 'hard_sigmoid'):
        x = (0.2 * x) + 0.5
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                            tf.cast(1., dtype=tf.float32))
        return x


def conv2d(name_or_scope, x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           weight_init=0.01, bias_init=0.1,
           regularizer=None,
           activation=lambda x : x,
           padding='SAME',
           device=None):
    """2D convolution that combines variable creation, activation
       and applying bias.
    Parameters
    ----------
    name_or_scope : str or VariableScope
        Variable scope to use.
    x : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    weight_init : float or function, optional
        Initialization's of the weights, either the standard deviation
        or a tensorflow initializer-fuction such as xavier init.
    bias_init: float or function or None, optional
        Whether to apply a constant bias (float) or a specific one (function)
        or none (None). 
    regularizer: (Tensor -> Tensor or None) function
        Regularizer function for the weight (not used for the bias).
        The result of applying it on a newly created variable will be added
        to the collection GraphKeys.REGULARIZATION_LOSSES and can be used
        for regularization.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    device: str or None, optional
        The device to which memory the variables will get stored on. (e.g. '/cpu:0')
    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(name_or_scope):
        if (isinstance(weight_init, types.FunctionType)):
            weight_init_func = weight_init
        elif (isinstance(weight_init, float)):
            weight_init_func = tf.truncated_normal_initializer(stddev=weight_init)
        else:
            raise ValueError("Parameter weight_init must be float or function.")
        
        w = get_variable(
            'W', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=weight_init_func,
            regularizer=regularizer,
            device=device)
        conv = tf.nn.conv2d(
            x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        
        if bias_init is not None:
            if (isinstance(bias_init, types.FunctionType)):
                bias_init_func = bias_init
            elif (isinstance(bias_init, float)):
                bias_init_func = tf.constant_initializer(bias_init)
            else:
                raise ValueError("Parameter bias_init must be float or function or None.")
            b = get_variable(
                'b', [n_filters],
                initializer=bias_init_func,
                device=device)
            linearity = tf.nn.bias_add(conv, b)
        else:
            linearity = conv
           
    return activation(linearity)


def conv2d_transpose(name_or_scope,
                     x, n_filters,
                     k_h=5, k_w=5,
                     stride_h=2, stride_w=2,
                     weight_init=0.01, bias_init=0.1,
                     regularizer=None,
                     activation=lambda x: x,
                     padding='SAME',
                     device=None):
    """2D transposed convolution (often called deconvolution, or upconvolution
       that combines variable creation, activation and applying bias.
    Parameters
    ----------
    name_or_scope : str or VariableScope
        Variable scope to use.
    x : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    weight_init : float or function, optional
        Initialization's of the weights, either the standard deviation
        or a tensorflow initializer-fuction such as xavier init.
    bias_init: float or function or None, optional
        Whether to apply a constant bias (float) or a specific one (function)
        or none (None). 
    regularizer: (Tensor -> Tensor or None) function
        Regularizer function for the weight (not used for the bias).
        The result of applying it on a newly created variable will be added
        to the collection GraphKeys.REGULARIZATION_LOSSES and can be used
        for regularization.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID' (currently only SAME is correctly implemented!)
    device: str or None, optional
        The device to which memory the variables will get stored on. (e.g. '/cpu:0')
    Returns
    -------
    x : Tensor
        Upconvolved input, which typically has a bigger size, but a lower depth.
    """
    with tf.variable_scope(name_or_scope):
        static_input_shape = x.get_shape().as_list()
        dyn_input_shape = tf.shape(x)
        
        # extract batch-size like as a symbolic tensor to allow variable size
        batch_size = dyn_input_shape[0]
        
        if (isinstance(weight_init, types.FunctionType)):
            weight_init_func = weight_init
        elif (isinstance(weight_init, float)):
            weight_init_func = tf.truncated_normal_initializer(stddev=weight_init)
        else:
            raise ValueError("Parameter weight_init must be float or function.")
        
        w = get_variable(
            'W', [k_h, k_w, n_filters, static_input_shape[3]],
            initializer=weight_init_func,
            regularizer=regularizer,
            device=device)
        
        assert padding in {'SAME', 'VALID'}
        if (padding is 'SAME'):
            out_h = dyn_input_shape[1] * stride_h
            out_w = dyn_input_shape[2] * stride_w
        elif (padding is 'VALID'):
            out_h = (dyn_input_shape[1] - 1) * stride_h + k_h
            out_w = (dyn_input_shape[2] - 1) * stride_w + k_w

        out_shape = tf.pack([batch_size, out_h, out_w, n_filters])
        
        convt = tf.nn.conv2d_transpose(
            x, w, output_shape=out_shape,
            strides=[1, stride_h, stride_w, 1], padding=padding)
        
        if bias_init is not None:
            if (isinstance(bias_init, types.FunctionType)):
                bias_init_func = bias_init
            elif (isinstance(bias_init, float)):
                bias_init_func = tf.constant_initializer(bias_init)
            else:
                raise ValueError("Parameter bias_init must be float or function or None.")
            b = get_variable(
                'b', [n_filters],
                initializer=bias_init_func,
                device=device)
            linearity = tf.nn.bias_add(convt, b)
        else:
            linearity = convt
    return activation(linearity)
    

def max_pool2d(x, k_h=5, k_w=5,
               stride_h=2, stride_w=2,
               padding='SAME',
               name=None):
    """2D max-pooling that combines variable creation, activation
       and applying bias.
    Parameters
    ----------
    x : Tensor
        Input tensor to convolve.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    stddev : float, optional
        Initialization's standard deviation.
    padding : str, optional
        'SAME' or 'VALID'
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Max-pooled input.
    """
    pooled = tf.nn.max_pool(
        x, ksize=[1, k_h, k_w, 1],
        strides=[1, stride_h, stride_w, 1],
        padding=padding, name=name)
    return pooled


def fc(name_or_scope, x, n_units,
       weight_init=0.01, bias_init=0.1,
       regularizer=None,
       activation=lambda x: x,
       device=None):
    """Fully-connected network .
    Parameters
    ----------
    name_or_scope : str or VariableScope
        Variable scope to use.
    x : Tensor
        Input tensor to the network.
    n_units : int
        Number of units to connect to.
    weight_init : float or function, optional
        Initialization's of the weights, either the standard deviation
        or a tensorflow initializer-fuction such as xavier init.
    bias_init: float or function or None, optional
        Whether to apply a constant bias (float) or a specific one (function)
        or none (None). 
    regularizer: (Tensor -> Tensor or None) function
        Regularizer function for the weight (not used for the bias).
        The result of applying it on a newly created variable will be added
        to the collection GraphKeys.REGULARIZATION_LOSSES and can be used
        for regularization.
    activation : arguments, optional
        Function which applies a nonlinearity
    device: str or None, optional
        The device to which memory the variables will get stored on. (e.g. '/cpu:0')
    Returns
    -------
    x : Tensor
        Fully-connected output.
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(name_or_scope):
        if (isinstance(weight_init, types.FunctionType)):
            weight_init_func = weight_init
        elif (isinstance(weight_init, float)):
            weight_init_func = tf.truncated_normal_initializer(stddev=weight_init)
        else:
            raise ValueError("Parameter weight_init must be float or function.")
        
        w = get_variable("W", [shape[1], n_units], tf.float32,
                            initializer=weight_init_func,
                            regularizer=regularizer,
                            device=device)
        mul = tf.matmul(x, w)
        
        if bias_init is not None:
            if (isinstance(bias_init, types.FunctionType)):
                bias_init_func = bias_init
            elif (isinstance(bias_init, float)):
                bias_init_func = tf.constant_initializer(bias_init)
            else:
                raise ValueError("Parameter bias_init must be float or function or None.")
            b = get_variable(
                'b', [n_units],
                initializer=bias_init_func,
                device=device)
            linearity = tf.nn.bias_add(mul, b)
        else:
            linearity = mul
    return activation(linearity)


# %%
def corrupt(x):
    """Take an input tensor and add uniform masking
    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

