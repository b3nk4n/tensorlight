import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensortools as tt
import base

from abc import ABCMeta


class MNISTBaseDataset(base.AbstractDataset):
    """MNIST base dataset wrapping the functions provided by tensorflow."""
    __metaclass__ = ABCMeta
    
    # load the dataset file lazily and just once
    _mnist = None
    
    """MNIST base dataset wrapping the functions provided by tensorflow."""
    def __init__(self, dataset):
        """Creates a dataset instance.
        Parameters
        ----------
        dataset: Dataset
            The TensorFlow MNIST dataset to use.
        """
        self._data = dataset.images.reshape((-1, 28, 28, 1))
        self._targets = dataset.labels
        dataset_size = dataset.num_examples
        
        self._indices = np.arange(dataset_size)
        self._row = 0
        
        super(MNISTBaseDataset, self).__init__(dataset_size, [28,28,1], [10])

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
    
    @tt.utils.attr.override
    def reset(self):
        self._row = 0
        np.random.shuffle(self._indices)
    
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
