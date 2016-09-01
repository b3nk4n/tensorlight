import numpy as np
import tensortools as tt

from abc import ABCMeta, abstractmethod, abstractproperty


class AbstractDataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_size):
        """Creates a dataset instance.
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        targets can be None: e.g. for image sequences or generated data
        """
        self._dataset_size = dataset_size
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
    
    @abstractproperty
    def input_shape(self):
        pass # TODO: implement it here from self._data? But can be overriden in sub class (moving mnist)
    
    @abstractproperty
    def target_shape(self):
        pass # TODO: implement it here from self._data? But can be overriden in sub class (moving mnist)
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_batch(self, batch_size):
        pass



class AbstractBatchDataset(AbstractDataset):
    __metaclass__ = ABCMeta

    def __init__(self, data, targets, dataset_size):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        targets can be None: e.g. for image sequences or generated data
        """
        self._data = data
        self._targets = targets
        
        self._indices = np.arange(dataset_size)
        self._row = 0
        super(AbstractBatchDataset, self).__init__(dataset_size)
    
    @tt.utils.attr.override
    def reset(self):
        self._row = 0
        np.random.shuffle(self._indices)
    
    
class AbstractImageBatchDataset(AbstractBatchDataset):
    __metaclass__ = ABCMeta

    def __init__(self, data, targets, dataset_size, image_size):
        assert len(image_size) == 3, "Image size has to have ndim=3."
        
        self._image_size = image_size
        super(AbstractImageBatchDataset, self).__init__(data, targets, dataset_size)
    
    @property
    @tt.utils.attr.override
    def input_shape(self):
        return [self._image_size[0], self._image_size[1], self._image_size[2]]
    
    @property
    @tt.utils.attr.override
    def target_shape(self):
        return [self._image_size[0], self._image_size[1], self._image_size[2]]
    
    @property 
    def image_size(self):
        return self._image_size
    
    
class AbstractImageSequenceBatchDataset(AbstractImageBatchDataset):
    __metaclass__ = ABCMeta

    def __init__(self, data, dataset_size, image_size,
                 input_seq_length, target_seq_length):
        self._input_seq_length = input_seq_length
        self._target_seq_length = target_seq_length
        super(AbstractImageSequenceBatchDataset, self).__init__(data, None, dataset_size, image_size) # Why-targets None?
    
    @property
    @tt.utils.attr.override
    def input_shape(self):
        return [self._input_seq_length] + super(AbstractImageSequenceBatchDataset, self).input_shape
    
    @property
    @tt.utils.attr.override
    def target_shape(self):
        return [self._target_seq_length] + super(AbstractImageSequenceBatchDataset, self).target_shape

    @property
    def input_seq_length(self):
        return self._input_seq_length
    
    @property
    def target_seq_length(self):
        return self._target_seq_length
    
    
    
class AbstractQueueDataset(AbstractDataset):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_size,
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
        super(AbstractQueueDataset, self).__init__(dataset_size)
            
    @tt.utils.attr.override
    def reset(self):
        return
    
    @property
    def min_examples_in_queue(self):
        return _self._min_examples_in_queue
    
    @property
    def queue_capacity(self):
        return self._queue_capacitiy
    
    @property
    def num_threads(self):
        return self._num_threads
    
    
class AbstractImageQueueDataset(AbstractQueueDataset):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_size, image_size,
                 min_examples_in_queue=256, queue_capacitiy=512, num_threads=8):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        targets can be None: e.g. for image sequences or generated data
        """
        assert len(image_size) == 3, "Image size has to have ndim=3."
        
        self._image_size = image_size
        super(AbstractImageQueueDataset, self).__init__(dataset_size, min_examples_in_queue, queue_capacitiy, num_threads)
            
    @property 
    def image_size(self):
        return self._image_size
    
    @property
    @tt.utils.attr.override
    def input_shape(self):
        return [self._image_size[0], self._image_size[1], self._image_size[2]]
    
    @property
    @tt.utils.attr.override
    def target_shape(self):
        return [self._image_size[0], self._image_size[1], self._image_size[2]]
    
    
class AbstractImageSequenceQueueDataset(AbstractImageQueueDataset):
    __metaclass__ = ABCMeta

    def __init__(self, dataset_size, image_size, input_seq_length, target_seq_length,
                 min_examples_in_queue=256, queue_capacitiy=512, num_threads=8):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        targets can be None: e.g. for image sequences or generated data
        """
        self._input_seq_length = input_seq_length
        self._target_seq_length = target_seq_length
        super(AbstractImageSequenceQueueDataset, self).__init__(dataset_size, image_size,
                                                                min_examples_in_queue, queue_capacitiy, num_threads)
            
    @property
    @tt.utils.attr.override
    def input_shape(self):
        return [self._input_seq_length] + super(AbstractImageSequenceQueueDataset, self).input_shape
    
    @property
    @tt.utils.attr.override
    def target_shape(self):
        return [self._target_seq_length] + super(AbstractImageSequenceQueueDataset, self).target_shape
            
    @property
    def input_seq_length(self):
        return self._input_seq_length
    
    @property
    def target_seq_length(self):
        return self._target_seq_length