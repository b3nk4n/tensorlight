import numpy as np
import tensortools as tt

from abc import ABCMeta, abstractmethod


class AbstractDataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_size, input_shape, target_shape):
        """Creates a dataset instance.
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        targets can be None: e.g. for image sequences or generated data
        """
        self._dataset_size = dataset_size
        self._input_shape = input_shape
        self._target_shape = target_shape
        self.reset()

    @property
    def size(self):
        return self._dataset_size
    
    @property
    def input_dims(self):
        return int(np.prod(self.input_shape))
    
    @property
    def target_dims(self):
        return int(np.prod(self.target_shape))
    
    @property
    def input_shape(self):
        return self._input_shape
    
    @property
    def target_shape(self):
        return self._target_shape
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_batch(self, batch_size):
        pass


    
class AbstractQueueDataset(AbstractDataset):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_size, input_shape, target_shape,
                 min_examples_in_queue=256, queue_capacitiy=512, num_threads=8):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        targets can be None: e.g. for image sequences or generated data
        """
        self._min_examples_in_queue = min_examples_in_queue
        self._queue_capacitiy = queue_capacitiy
        self._num_threads = num_threads
        super(AbstractQueueDataset, self).__init__(dataset_size, input_shape, target_shape)
            
    @tt.utils.attr.override
    def reset(self):
        return
    
    @property
    def min_examples_in_queue(self):
        return self._min_examples_in_queue
    
    @property
    def queue_capacity(self):
        return self._queue_capacitiy
    
    @property
    def num_threads(self):
        return self._num_threads
