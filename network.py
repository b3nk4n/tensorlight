import tensorflow as tf

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
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           bias=0.1,
           activation=lambda x: x,
           padding='SAME',
           name=None):
    """2D convolution that combines variable creation, activation
    and applying bias.
    Parameters
    ----------
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
    stddev : float, optional
        Initialization's standard deviation.
    bias: float, optional
        Whether to apply a bias or not.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            'W', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        b = tf.get_variable(
            'b', [n_filters],
            initializer=tf.constant_initializer(bias))
        return activation(conv + b)


def conv2d_transpose(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           bias=0.1,
           activation=lambda x: x,
           padding='SAME',
           name=None):
    """2D transposed convolution (often called deconvolution, or upconvolution
    that combines variable creation, activation and applying bias.
    Parameters
    ----------
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
    stddev : float, optional
        Initialization's standard deviation.
    bias: float, optional
        Whether to apply a bias or not.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID' (currently only SAME is correctly implemented!)
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Upconvolved input, which typically has a bigger size, but a lower depth.
    """
    with tf.variable_scope(name):
        input_shape = x.get_shape()
        w = tf.get_variable(
            'W', [k_h, k_w, n_filters, input_shape[3]],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d_transpose(
            x, w, output_shape=[input_shape[0], input_shape[1] * stride_h, input_shape[2] * stride_w, n_filters],
            strides=[1, stride_h, stride_w, 1], padding=padding)
        b = tf.get_variable(
            'b', [n_filters],
            initializer=tf.constant_initializer(bias))
        return activation(conv + b)
    

def max_pool(x, k_h=5, k_w=5,
            stride_h=2, stride_w=2,
            padding='SAME',
            name=None):
    """Max-Pooling that combines variable creation, activation
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
    with tf.variable_scope(name):
        pooled = tf.nn.max_pool(
            x, ksize=[1, k_h, k_w, 1],
            strides=[1, stride_h, stride_w, 1],
            padding=padding)
        return pooled


def linear(x, n_units, name=None, stddev=0.02, bias=0.1,
           activation=lambda x: x):
    """Fully-connected network .
    Parameters
    ----------
    x : Tensor
        Input tensor to the network.
    n_units : int
        Number of units to connect to.
    name : str, optional
        Variable scope to use.
    stddev : float, optional
        Initialization's standard deviation.
    bias: float, optional
        Whether to apply a bias or not.
    activation : arguments, optional
        Function which applies a nonlinearity
    Returns
    -------
    x : Tensor
        Fully-connected output.
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable("W", [shape[1], n_units], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable(
            'b', [n_units],
            initializer=tf.constant_initializer(bias))
        return activation(tf.matmul(x, matrix))


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


# DEPRECATED: Offers low flexibility. Use tf.get_variable() instead.
def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


# DEPRECATED: Offers low flexibility. Use tf.get_variable() instead.
def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.constant(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

