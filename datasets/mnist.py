import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensortools as tt
import base


class MNISTTrainDataset(base.AbstractImageDataset):
    """MNIST training dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        """Creates a dataset instance."""
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        data = mnist.train.images.reshape((-1, 28, 28, 1))
        targets = mnist.train.labels
        dataset_size = mnist.train.num_examples
        
        super(MNISTTrainDataset, self).__init__(batch_size, data, targets, dataset_size, (28, 28, 1))

    @tt.utils.attr.override
    def get_batch(self):
        if self._row + self.batch_size >= self._data.shape[0]:
            self.reset()
        start = self._row
        end = start + self.batch_size
        ind_range = self._indices[start:end]
        self._row += self.batch_size
        images = self._data[ind_range]
        labels = self._targets[ind_range]
        return images, labels
        
        
class MNISTValidDataset(base.AbstractImageDataset):
    """MNIST validation dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        """Creates a dataset instance."""
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        data = mnist.validation.images.reshape((-1, 28, 28, 1))
        targets = mnist.validation.labels
        dataset_size = mnist.validation.num_examples
        
        super(MNISTValidDataset, self).__init__(batch_size, data, targets, dataset_size, (28, 28, 1))

    @tt.utils.attr.override
    def get_batch(self):
        if self._row + self.batch_size >= self._data.shape[0]:
            self.reset()
        start = self._row
        end = start + self.batch_size
        ind_range = self._indices[start:end]
        self._row += self.batch_size
        images = self._data[ind_range]
        labels = self._targets[ind_range]
        return images, labels
    
    
class MNISTTestDataset(base.AbstractImageDataset):
    """MNIST test dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        """Creates a dataset instance."""
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        data = mnist.test.images.reshape((-1, 28, 28, 1))
        targets = mnist.test.labels
        dataset_size = mnist.test.num_examples
        
        super(MNISTValidDataset, self).__init__(batch_size, data, targets, dataset_size, (28, 28, 1))

    @tt.utils.attr.override
    def get_batch(self):
        if self._row + self.batch_size >= self._data.shape[0]:
            self.reset()
        start = self._row
        end = start + self.batch_size
        ind_range = self._indices[start:end]
        self._row += self.batch_size
        images = self._data[ind_range]
        labels = self._targets[ind_range]
        return images, labels