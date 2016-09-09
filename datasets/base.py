import numpy as np
import tensortools as tt

from abc import ABCMeta, abstractmethod


class AbstractDataset(object):
    """Dataset base class, mainly used for datasets with feeding."""
    __metaclass__ = ABCMeta

    def __init__(self, data_dir, dataset_size, input_shape, target_shape):
        """Creates a dataset instance.
        Parameters
        ----------
        data_dir: str
            The data directory where the data will be stored.
        dataset_size: int
            The dataset site.
        input_shape: list(int)
            The shape of the inputs.
        target_shape: list(int)
            The shape of the targets.
        """
        self._data_dir = data_dir
        self._dataset_size = dataset_size
        self._input_shape = input_shape
        self._target_shape = target_shape
        self.reset()
        
    @abstractmethod
    def reset(self):
        """Resets the dataset."""
        pass

    @abstractmethod
    def get_batch(self, batch_size):
        """Gets the next batch.
        Parameters
        ----------
        batch_size: int
            The size of the next batch.
        Returns
        ----------
        The next batch tuple (input, target) with shape size
        'input_shape' and 'target_shape'.
        """
        pass
    
    @property
    def data_dir(self):
        """Gets the data directory."""
        return self._data_dir

    @property
    def size(self):
        """Gets the dataset size as int."""
        return self._dataset_size
    
    @property
    def input_dims(self):
        """Gets the total input dimensions."""
        return int(np.prod(self.input_shape))
    
    @property
    def target_dims(self):
        """Gets the total target dimensions."""
        return int(np.prod(self.target_shape))
    
    @property
    def input_shape(self):
        """Gets the input shape."""
        return self._input_shape
    
    @property
    def target_shape(self):
        """Gets the target shape."""
        return self._target_shape


    
class AbstractQueueDataset(AbstractDataset):
    """Dataset base class, used for datasets with input queue."""
    __metaclass__ = ABCMeta

    def __init__(self, data_dir, dataset_size, input_shape, target_shape,
                 min_examples_in_queue=512, queue_capacitiy=1024, num_threads=8):
        """Creates a dataset instance that uses a queue.
        Parameters
        ----------
        data_dir: str
            The data directory where the data will be stored.
        dataset_size: int
            The dataset site.
        input_shape: list(int)
            The shape of the inputs.
        target_shape: list(int)
            The shape of the targets.
        min_examples_in_queue: int, optional
            The minimum examples that have to be in the queue.
            A higher value ensures a good mix.
        queue_capacitiy: int, optional
            The maximum capacity of the input queue.
        num_threads: int, optional
            The number of threads to generate the inputs.
        """
        self._min_examples_in_queue = min_examples_in_queue
        self._queue_capacitiy = queue_capacitiy
        self._num_threads = num_threads
        super(AbstractQueueDataset, self).__init__(data_dir, dataset_size, input_shape, target_shape)
            
    @tt.utils.attr.override
    def reset(self):
        # Usually No-Op, because re-shuffling is performed by the queue itself.
        return
    
    @property
    def min_examples_in_queue(self):
        """Gets the minimum number of examples in the queue."""
        return self._min_examples_in_queue
    
    @property
    def queue_capacity(self):
        """Gets the queue capacity."""
        return self._queue_capacitiy
    
    @property
    def num_threads(self):
        """Gets the number of producer threads."""
        return self._num_threads
