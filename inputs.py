import numpy as np
import tensorflow as tf


def generate_batch(inputs, target, batch_size, min_queue_examples,
                   shuffle=True, num_threads=8):
    """Construct a queued batch of data (e.g images) and labels.
    Parameters
    ----------
    inputs: n-D Tensor
        The input to generate an input batch from.
    target:  n-D Tensor
        The target to of the provided input.
    batch_size: int
        Number of data examples per batch.
    min_queue_examples: int
        Minimum number of samples to retain in the queue
        that provides of batches of examples. We also keep
    shuffle: boolean, optional
        Indicates whether to use a shuffling queue.
    num_threads: int
        The number of threads to read inputs from the queue.
        Take care that a value >1 (default) could lead to race-conditions
        in data preprocessing, when there are control-flow dependencies with
        random operations.
    Returns
    ----------
    images: 4-D Tensor of shape [batch_size, height, width, channels]
        Images. 4D tensor of  size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' many images and labels from the example queue
    if shuffle:
        inputs_batch, target_batch = tf.train.shuffle_batch(
            [inputs, target],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        inputs_batch, target_batch = tf.train.batch(
            [inputs, target],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return inputs_batch, target_batch