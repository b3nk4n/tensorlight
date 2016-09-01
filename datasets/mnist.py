import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensortools as tt
import base

from abc import ABCMeta


class MNISTBaseDataset(base.AbstractImageBatchDataset):
    __metaclass__ = ABCMeta
    
    # load the dataset file lazily and just once
    _mnist = None
    
    """MNIST base dataset wrapping the functions provided by tensorflow."""
    def __init__(self, dataset):
        """Creates a dataset instance."""
        
        data = dataset.images.reshape((-1, 28, 28, 1))
        targets = dataset.labels
        dataset_size = dataset.num_examples
        super(MNISTBaseDataset, self).__init__(data, targets, dataset_size, (28, 28, 1))

    @tt.utils.attr.override
    def get_batch(self, batch_size):
        if self._row + batch_size >= self._data.shape[0]:
            self.reset()
        start = self._row
        end = start + batch_size
        ind_range = self._indices[start:end]
        self._row += batch_size
        images = self._data[ind_range]
        labels = self._targets[ind_range]
        return images, labels
    
    @staticmethod
    def mnist():
        if MNISTBaseDataset._mnist is None:
            MNISTBaseDataset._mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        return MNISTBaseDataset._mnist



class MNISTTrainDataset(MNISTBaseDataset):
    """MNIST training dataset wrapping the functions provided by tensorflow."""
    def __init__(self):
        mnist = MNISTBaseDataset.mnist()
        super(MNISTTrainDataset, self).__init__(mnist.train)
        
        
class MNISTValidDataset(MNISTBaseDataset):
    """MNIST validation dataset wrapping the functions provided by tensorflow."""
    def __init__(self):
        mnist = MNISTBaseDataset.mnist()
        super(MNISTValidDataset, self).__init__(mnist.validation)

    
class MNISTTestDataset(MNISTBaseDataset):
    """MNIST test dataset wrapping the functions provided by tensorflow."""
    def __init__(self):
        mnist = MNISTBaseDataset.mnist()
        super(MNISTTestDataset, self).__init__(mnist.test)
