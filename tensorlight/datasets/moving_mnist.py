import sys
from abc import ABCMeta

import h5py
import numpy as np

import tensorlight as light
import base


# we use the same MNIST dataset as University of Toronto in its 'Unsupervised Learning with LSTMS'
# paper to make sure we can compare our results with theirs.
MNIST_URL = 'http://www.cs.toronto.edu/~emansim/datasets/mnist.h5'
MNIST_TEST_URL = 'http://www.cs.toronto.edu/~emansim/datasets/bouncing_mnist_test.npy'


class MovingMNISTBaseGeneratedDataset(base.AbstractDataset):
    """Moving MNIST dataset.
       Reference: Based on Srivastava et al.
                  http://www.cs.toronto.edu/~nitish/unsupervised_video/
    """
    __metaclass__ = ABCMeta
    
    """Moving MNIST dataset that creates data on the fly."""
    def __init__(self, dataset_key, data_dir, dataset_size, input_shape=[10, 64, 64, 1],
                 target_shape=[10, 64, 64, 1], as_binary=False, num_digits=2, step_length=0.1):
        """Creates a base MovingMNIST dataset instance.
        Parameters
        ----------
        dataset_key: str
            The dataset dictionary key.
        data_dir: str
            The path where the data will be stored.
        dataset_size: int
            The dataset size.
        input_shape: list(int) of shape [t, h, w, c]
            The input image sequence shape.
        target_shape: list(int) of shape [t, h, w, c]
            The taget image sequence shape.
        as_binary: Boolean, optional
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
        num_digits: int, optional
            The number of flying MNIST digits.
        step_length: float, optional
            The step length of movement per frame.
        """
        assert input_shape[1:] == target_shape[1:], "Image data shapes have to be equal."
        assert len(input_shape) == 4, "Input and target shapes require ndims == 4."
        
        self._num_digits = num_digits
        self._step_length = step_length
        self._digit_size = 28
        
        try:
            filepath = light.utils.data.download(MNIST_URL, data_dir)
            f = h5py.File(filepath)
            data = f[dataset_key].value
        except:
            print 'Please set the correct path to MNIST dataset. Might be caused by a download error.'
            sys.exit()
        f.close()
        
        data = data.reshape(-1, self._digit_size, self._digit_size)
        
        if as_binary:
            self._data = light.utils.data.as_binary(data)
        else:
            self._data = data
        
        # here: the indices/rows are used for the internal MNIST data 
        self._indices = np.arange(self._data.shape[0])
        self._row = 0
        
        super(MovingMNISTBaseGeneratedDataset, self).__init__(data_dir, dataset_size, input_shape, target_shape)
    
    @light.utils.attr.override
    def get_batch(self, batch_size):
        input_seq_length = self.input_shape[0]
        target_seq_length = self.target_shape[0]
        total_seq_length = input_seq_length + target_seq_length
        start_y, start_x = MovingMNISTBaseGeneratedDataset._get_random_trajectory(batch_size * self._num_digits,
                                                                                  total_seq_length,
                                                                                  self.input_shape[1:],
                                                                                  self._digit_size,
                                                                                  self._step_length)
    
        input_data = np.zeros([batch_size] + self.input_shape, dtype=np.float32)
        
        target_data = None
        if target_seq_length > 0:
            target_data = np.zeros([batch_size] + self.target_shape, dtype=np.float32)
    
        for j in xrange(batch_size):
            for n in xrange(self._num_digits):
       
                # get random digit from dataset
                ind = self._indices[self._row]
                self._row += 1
                if self._row == self._data.shape[0]:
                    self.reset()
                digit_image = self._data[ind, :, :]
        
                # generate inputs
                for i in xrange(input_seq_length):
                    top    = start_y[i, j * self._num_digits + n]
                    left   = start_x[i, j * self._num_digits + n]
                    bottom = top  + self._digit_size
                    right  = left + self._digit_size
                    # set data and use maximum for overlap
                    input_data[j, i, top:bottom, left:right, 0] = np.maximum(input_data[j, i, top:bottom, left:right, 0],
                                                                             digit_image)
                # generate targets
                offset = input_seq_length
                for i in xrange(target_seq_length):
                    top    = start_y[i + offset, j * self._num_digits + n]
                    left   = start_x[i + offset, j * self._num_digits + n]
                    bottom = top  + self._digit_size
                    right  = left + self._digit_size
                    # set data and use maximum for overlap
                    target_data[j, i, top:bottom, left:right, 0] = np.maximum(target_data[j, i, top:bottom, left:right, 0],
                                                                              digit_image)
        return input_data, target_data
    
    @light.utils.attr.override
    def reset(self):
        self._row = 0
        np.random.shuffle(self._indices)
    
    @staticmethod
    def _get_random_trajectory(batch_size, length, image_size, digit_size, step_length):
        canvas_size_h = image_size[0] - digit_size
        canvas_size_w = image_size[1] - digit_size

        # Initial position uniform random inside the box.
        y = np.random.rand(batch_size)
        x = np.random.rand(batch_size)

        # Choose a random velocity.
        theta = np.random.rand(batch_size) * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros((length, batch_size))
        start_x = np.zeros((length, batch_size))
        for i in xrange(length):
            # Take a step along velocity.
            y += v_y * step_length
            x += v_x * step_length

            # Bounce off edges.
            for j in xrange(batch_size):
                if x[j] <= 0:
                    x[j] = 0
                    v_x[j] = -v_x[j]
                if x[j] >= 1.0:
                    x[j] = 1.0
                    v_x[j] = -v_x[j]
                if y[j] <= 0:
                    y[j] = 0
                    v_y[j] = -v_y[j]
                if y[j] >= 1.0:
                    y[j] = 1.0
                    v_y[j] = -v_y[j]
            start_y[i, :] = y
            start_x[i, :] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size_h * start_y).astype(np.int32)
        start_x = (canvas_size_w * start_x).astype(np.int32)
        return start_y, start_x



class MovingMNISTTrainDataset(MovingMNISTBaseGeneratedDataset):
    """Moving MNIST train dataset that creates data on the fly."""
    def __init__(self, data_dir, input_shape=[10, 64, 64, 1], target_shape=[10, 64, 64, 1],
                 as_binary=False, num_digits=2, step_length=0.1):
        """Creates a traning MovingMNIST dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_shape: list(int) of shape [t, h, w, c]
            The input image sequence shape.
        target_shape: list(int) of shape [t, h, w, c]
            The taget image sequence shape.
        as_binary: Boolean, optional
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
        num_digits: int, optional
            The number of flying MNIST digits.
        step_length: float, optional
            The step length of movement per frame.
        """
        dataset_size = sys.maxint
        super(MovingMNISTTrainDataset, self).__init__('train', data_dir, dataset_size,
                                                      input_shape, target_shape,
                                                      as_binary, num_digits, step_length)
    
    
    
class MovingMNISTValidDataset(MovingMNISTBaseGeneratedDataset):
    """Moving MNIST validation dataset that creates data on the fly."""
    def __init__(self, data_dir, input_shape=[10, 64, 64, 1], target_shape=[10, 64, 64, 1],
                 as_binary=False, num_digits=2, step_length=0.1):
        """Creates a validation MovingMNIST dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_shape: list(int) of shape [t, h, w, c]
            The input image sequence shape.
        target_shape: list(int) of shape [t, h, w, c]
            The taget image sequence shape.
        as_binary: Boolean, optional
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
        num_digits: int, optional
            The number of flying MNIST digits.
        step_length: float, optional
            The step length of movement per frame.
        """
        dataset_size = 10000
        super(MovingMNISTValidDataset, self).__init__('validation', data_dir, dataset_size,
                                                      input_shape, target_shape,
                                                      as_binary, num_digits, step_length)

    
    
class MovingMNISTTestDataset(base.AbstractDataset):
    """Moving MNIST test dataset that that uses the same data as in other papers."""
    def __init__(self, data_dir, input_seq_length=10, target_seq_length=10, as_binary=False):
        """Creates a test MovingMNIST dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_seq_length: int, optional
            The input sequence length
        target_seq_length: int, optional
            The target sequence length
        as_binary: Boolean, optional
            Whether the data should be returned as float (default) in range
            [0.0, 1.0], or as pseudo-binary with values {0.0, 1.0}, were the
            original data.
        """
        assert input_seq_length + target_seq_length <= 20, "The maximum total test sequence length is 20."
        
        try:
            filepath = light.utils.data.download(MNIST_TEST_URL, data_dir)
            print("Loading MNIST test set from numpy-array. This might take a while...")
            data = np.load(filepath)
            data = np.float32(data)
        except:
            print 'Please set the correct path to the dataset. Might be caused by a download error.'
            sys.exit()

        # introduce channel dimension
        data = np.expand_dims(data, axis=4)
        
        # use value scale [0,1]
        data = data / 255.0 
        
        if as_binary:
            self._data = light.utils.data.as_binary(data)
        else:
            self._data = data
        
        dataset_size = data.shape[0]
        self._row = 0
        
        super(MovingMNISTTestDataset, self).__init__(data_dir, dataset_size, input_shape=[input_seq_length, 64, 64, 1],
                                                     target_shape=[target_seq_length, 64, 64, 1])
    @light.utils.attr.override
    def get_batch(self, batch_size):
        if self._row + batch_size >= self.size:
            self.reset()

        batch_inputs = self._data[self._row:self._row+batch_size,
                                  0:self.input_shape[0],:,:,:]
        batch_targets = self._data[self._row:self._row+batch_size,
                                   self.input_shape[0]:self.input_shape[0]+self.target_shape[0],:,:,:]
        self._row = self._row + batch_size
        return batch_inputs, batch_targets
    
    @light.utils.attr.override
    def reset(self):
        self._row = 0