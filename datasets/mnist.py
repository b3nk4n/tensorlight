import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensortools as tt
import base


class MNISTTrainDataset(base.AbstractImageDataset):
    """MNIST dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        """Creates a dataset instance."""
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        data = mnist.train       
        dataset_size = data.num_examples
        
        super(MNISTTrainDataset, self).__init__(batch_size, data, dataset_size, (28, 28, 1))

    @tt.utils.attr.override
    def get_batch(self):
        batch_x, batch_y = self.data.train.next_batch(self.batch_size)
        batch_x = batch_x.reshape((-1, 28, 28, 1))
        return batch_x, batch_y
        
        
class MNISTValidDataset(base.AbstractImageDataset):
    """MNIST dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        """Creates a dataset instance."""
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        data = mnist.valid        
        dataset_size = data.num_examples
        
        super(MNISTValidDataset, self).__init__(batch_size, data, dataset_size, (28, 28, 1))

    @tt.utils.attr.override
    def get_batch(self):
        batch_x, batch_y = self.data.valid.next_batch(self.batch_size)
        batch_x = batch_x.reshape((-1, 28, 28, 1))
        return batch_x, batch_y