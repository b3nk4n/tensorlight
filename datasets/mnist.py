import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensortools as tt
import base

from abc import ABCMeta


class MNISTBaseDataset(base.AbstractImageDataset):
    __metaclass__ = ABCMeta
    
    """MNIST base dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size, dataset):
        """Creates a dataset instance."""
        data = dataset.images.reshape((-1, 28, 28, 1))
        targets = dataset.labels
        dataset_size = dataset.num_examples
        super(MNISTBaseDataset, self).__init__(batch_size, data, targets, dataset_size, (28, 28, 1))

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


class MNISTTrainDataset(MNISTBaseDataset):
    """MNIST training dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        super(MNISTTrainDataset, self).__init__(batch_size, mnist.train)
        
        
class MNISTValidDataset(MNISTBaseDataset):
    """MNIST validation dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        super(MNISTValidDataset, self).__init__(batch_size, mnist.validation)

    
class MNISTTestDataset(MNISTBaseDataset):
    """MNIST test dataset wrapping the functions provided by tensorflow."""
    def __init__(self, batch_size):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        super(MNISTTestDataset, self).__init__(batch_size, mnist.test)
