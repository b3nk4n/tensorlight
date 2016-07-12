import math

import numpy as np
import tensorflow as tf


def identity_initializer(scale = 1.0):
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


def orthogonal_initializer(scale = 1.1):
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
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer
