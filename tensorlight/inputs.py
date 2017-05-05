import numpy as np
import tensorflow as tf


def generate_batch(inputs, target, batch_size, min_queue_examples,
                   queue_capacitiy, shuffle=True, num_threads=8):
    """Construct a queued batch of data (e.g images) and labels.
    Parameters
    ----------
    inputs: n-D Tensor
        The input to generate an input batch from.
    target:  n-D Tensor
        The target to of the provided input.
    batch_size: int or Tensor/Placeholder
        Number of data examples per batch.
    min_queue_examples: int
        Minimum number of samples to retain in the queue
        that provides of batches of examples.
    queue_capacity: int
        The capacity of the queue, which must be higher than min_queue_examples.
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
    assert min_queue_examples < queue_capacitiy, "The value of 'min_queue_examples' has to be lower than 'queue_capacitiy'."
    
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' many images and labels from the example queue
    if shuffle:
        inputs_batch, target_batch = tf.train.shuffle_batch(
            [inputs, target],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=queue_capacitiy,
            min_after_dequeue=min_queue_examples)
    else:
        inputs_batch, target_batch = tf.train.batch(
            [inputs, target],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=queue_capacitiy)

    return inputs_batch, target_batch