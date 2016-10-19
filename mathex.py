import tensorflow as tf


def log10(value):
    """Calculates the base-10 log of each element.
    Parameters
    ----------
    value: Tensor
        The tensor from which to calculate the base-10 log.
    Returns
    ----------
    Calculated tensor with the base-10 log of each element.
    """
    with tf.name_scope("log_10"):
        numerator = tf.log(value)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator