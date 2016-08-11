import math

import numpy as np
import tensorflow as tf


def identity_initializer(scale=1.0):
    """Identity initializer by Quoc V. Le et al.
    This is also recommended by at least one paper to initialize
    the weights matrix in a RNN.
    References:
        Paper: Quoc V. Le et al., http://arxiv.org/abs/1504.00941
    Parameters
    ----------
    scale: float, optional
        The scale of the indentity values.
    Returns
    ----------
    _initializer: function
        Returns the init function.
    """
    def _initializer(shape, dtype=tf.float32):
        if len(shape) == 1:
            return tf.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(scale*np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(scale*array, dtype=dtype)
        else:
            raise ValueError("Invalid shape.")
    return _initializer


def _orthonogal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)  #this needs to be corrected to float32


def orthogonal_initializer(scale=1.0):
    """Orthogonal initializer by Saxe et al.
       This initialization is recommended for initializing the
       hidden weights in a RNN.
    References:
        From Lasagne and Keras.
        Paper: Saxe et al., http://arxiv.org/abs/1312.6120
    Parameters
    ----------
    scale: float, optional
        The scale of the orthogonal values.
    Returns
    ----------
    _initializer: function
        Returns the init function.
    """
    def _initializer(shape, dtype=tf.float32):
        q = _orthogonal(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


def bn_lstm_identity_initializer(scale=1.0):
    """Special indentity initializer used for batch normalization in LSTMs.
    References:
        From: http://olavnymoen.com/2016/07/07/rnn-batch-normalization
    Parameters
    ----------
    scale: float, optional
        The scale of the identity values.
    Returns
    ----------
    _initializer: function
        Returns the init function.
    """
    def _initializer(shape, dtype=tf.float32):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale  # j
        t[:, :size] = _orthonogal([size, size])  # i
        t[:, size * 2:size * 3] = _orthonogal([size, size])  # f
        t[:, size * 3:] = _orthonogal([size, size])  # o
        return tf.constant(t, dtype)

    return _initializer