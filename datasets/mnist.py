import os
from abc import ABCMeta

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import tensorlight as light
import base


class MNISTBaseDataset(base.AbstractDataset):
    """MNIST base dataset wrapping the functions provided by tensorflow."""
    __metaclass__ = ABCMeta
    
    # load the dataset file lazily and just once
    _mnist = None
    
    """MNIST base dataset wrapping the functions provided by tensorflow."""
    def __init__(self, data_dir, dataset, as_binary):
        """Creates a dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        dataset: Dataset
            The TensorFlow MNIST dataset to use.
        as_binary: Boolean
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
        """
        data = dataset.images.reshape((-1, 28, 28, 1))
        if as_binary:
            self._data = light.utils.data.as_binary(data)
        else:
            self._data = data
        
        self._targets = dataset.labels
        dataset_size = dataset.num_examples
        
        self._indices = np.arange(dataset_size)
        self._row = 0
        
        super(MNISTBaseDataset, self).__init__(data_dir, dataset_size, [28,28,1], [10])

    @light.utils.attr.override
    def get_batch(self, batch_size):
        if self._row + batch_size >= self.size:
            self.reset()
        start = self._row
        end = start + batch_size
        ind_range = self._indices[start:end]
        self._row += batch_size
        images = self._data[ind_range]
        labels = self._targets[ind_range]
        return images, labels
    
    @light.utils.attr.override
    def reset(self):
        self._row = 0
        np.random.shuffle(self._indices)
    
    @staticmethod
    def mnist(data_dir):
        if MNISTBaseDataset._mnist is None:
            directory = os.path.join(data_dir, 'MNIST_data')
            MNISTBaseDataset._mnist = input_data.read_data_sets(directory, one_hot=True)
        return MNISTBaseDataset._mnist



class MNISTTrainDataset(MNISTBaseDataset):
    """MNIST training dataset wrapping the functions provided by tensorflow.
    Parameters
    ----------
    data_dir: str
            The path where the data will be stored.
    as_binary: Boolean
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
    """
    def __init__(self, data_dir, as_binary=False):
        mnist = MNISTBaseDataset.mnist(data_dir)
        super(MNISTTrainDataset, self).__init__(data_dir, mnist.train, as_binary)
        
        
class MNISTValidDataset(MNISTBaseDataset):
    """MNIST validation dataset wrapping the functions provided by tensorflow.
    Parameters
    ----------
    data_dir: str
            The path where the data will be stored.
    as_binary: Boolean
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
    """
    def __init__(self, data_dir, as_binary=False):
        mnist = MNISTBaseDataset.mnist(data_dir)
        super(MNISTValidDataset, self).__init__(data_dir, mnist.validation, as_binary)

    
class MNISTTestDataset(MNISTBaseDataset):
    """MNIST test dataset wrapping the functions provided by tensorflow.
    Parameters
    ----------
    data_dir: str
            The path where the data will be stored.
    as_binary: Boolean
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
    """
    def __init__(self, data_dir, as_binary=False):
        mnist = MNISTBaseDataset.mnist(data_dir)
        super(MNISTTestDataset, self).__init__(data_dir, mnist.test, as_binary)
