import sys

import h5py
import numpy as np
import tensortools as tt
import base


# we use the same MNIST dataset as University of Toronto in its 'Unsupervised Learning with LSTMS'
# paper to make sure we can compare our results with theirs.
MNIST_URL = 'http://www.cs.toronto.edu/~emansim/datasets/mnist.h5'
MNIST_TEST_URL = 'http://www.cs.toronto.edu/~emansim/datasets/bouncing_mnist_test.npy'


def _get_random_trajectory(batch_size, seq_length, image_size, digit_size, step_length):
    length = seq_length
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


class MovingMNISTTrainDataset(base.AbstractImageSequenceDataset):
    """Moving MNIST dataset that creates data on the fly."""
    def __init__(self, batch_size, image_size=(64, 64, 1), input_seq_length=10, target_seq_length=10,
                 num_digits=2, step_length=0.1):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        """
        self._num_digits = num_digits
        self._step_length = step_length
        
        try:
            filepath = tt.utils.data.download(MNIST_URL, 'tmp')
            f = h5py.File(filepath)
            data = f['train'].value
        except:
            print 'Please set the correct path to MNIST dataset. Might be caused by a download error.'
            sys.exit()
        f.close()
        
        data = data.reshape(-1, 28, 28)
        dataset_size = sys.maxint
        
        super(MovingMNISTTrainDataset, self).__init__(batch_size, data, dataset_size, image_size,
                                                      input_seq_length, target_seq_length)

    @tt.utils.attr.override
    def get_batch(self):
        start_y, start_x = _get_random_trajectory(self._batch_size * self._num_digits,
                                                  self.input_seq_length + self.target_seq_length, 
                                                  self._image_size, 
                                                  28, 
                                                  self._step_length)
    
        input_data = np.zeros([self._batch_size] + self.input_shape, dtype=np.float32)
        
        target_data = None
        if self.target_seq_length > 0:
            target_data = np.zeros([self._batch_size] + self.target_shape, dtype=np.float32)
    
        for j in xrange(self._batch_size):
            for n in xrange(self._num_digits):
       
                # get random digit from dataset
                ind = self._indices[self._row]
                self._row += 1
                if self._row == self._data.shape[0]:
                    self.reset()
                digit_image = self._data[ind, :, :]
        
                # generate inputs
                for i in xrange(self.input_seq_length):
                    top    = start_y[i, j * self._num_digits + n]
                    left   = start_x[i, j * self._num_digits + n]
                    bottom = top  + 28
                    right  = left + 28
                    # set data and use maximum for overlap
                    input_data[j, i, top:bottom, left:right, 0] = np.maximum(input_data[j, i, top:bottom, left:right, 0],
                                                                             digit_image)
                # generate targets
                for i in xrange(self.target_seq_length):
                    top    = start_y[i, j * self._num_digits + n]
                    left   = start_x[i, j * self._num_digits + n]
                    bottom = top  + 28
                    right  = left + 28
                    # set data and use maximum for overlap
                    target_data[j, i, top:bottom, left:right, 0] = np.maximum(target_data[j, i, top:bottom, left:right, 0],
                                                                              digit_image)
        return input_data, target_data
    
    
    
class MovingMNISTValidDataset(base.AbstractImageSequenceDataset):
    """Moving MNIST dataset that creates data on the fly."""
    def __init__(self, batch_size, image_size=(64, 64, 1), input_seq_length=10, target_seq_length=10,
                 num_digits=2, step_length=0.1):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        """
        self._num_digits = num_digits
        self._step_length = step_length
        
        try:
            filepath = tt.utils.data.download(MNIST_URL, 'tmp')
            f = h5py.File(filepath)
            data = f['validation'].value
        except:
            print 'Please set the correct path to MNIST dataset. Might be caused by a download error.'
            sys.exit()
        f.close()

        data = data.reshape(-1, 28, 28)
        dataset_size = 10000
        
        super(MovingMNISTValidDataset, self).__init__(batch_size, data, dataset_size, image_size,
                                                      input_seq_length, target_seq_length)

    @tt.utils.attr.override
    def get_batch(self):
        start_y, start_x = _get_random_trajectory(self._batch_size * self._num_digits,
                                                  self.input_seq_length + self.target_seq_length, 
                                                  self._image_size, 
                                                  28, 
                                                  self._step_length)
    
        input_data = np.zeros([self._batch_size] + self.input_shape, dtype=np.float32)
        
        target_data = None
        if self.target_seq_length > 0:
            target_data = np.zeros([self._batch_size] + self.target_shape, dtype=np.float32)
    
        for j in xrange(self._batch_size):
            for n in xrange(self._num_digits):
       
                # get random digit from dataset
                ind = self._indices[self._row]
                self._row += 1
                if self._row == self._data.shape[0]:
                    self.reset()
                digit_image = self._data[ind, :, :]
        
                # generate inputs
                for i in xrange(self.input_seq_length):
                    top    = start_y[i, j * self._num_digits + n]
                    left   = start_x[i, j * self._num_digits + n]
                    bottom = top  + 28
                    right  = left + 28
                    # set data and use maximum for overlap
                    input_data[j, i, top:bottom, left:right, 0] = np.maximum(input_data[j, i, top:bottom, left:right, 0],
                                                                             digit_image)
                # generate targets
                for i in xrange(self.target_seq_length):
                    top    = start_y[i, j * self._num_digits + n]
                    left   = start_x[i, j * self._num_digits + n]
                    bottom = top  + 28
                    right  = left + 28
                    # set data and use maximum for overlap
                    target_data[j, i, top:bottom, left:right, 0] = np.maximum(target_data[j, i, top:bottom, left:right, 0],
                                                                              digit_image)
        return input_data, target_data

    
    
# video patches loaded from some file
class MovingMNISTTestDataset(base.AbstractImageSequenceDataset):
    def __init__(self, batch_size, input_seq_length=10, target_seq_length=10):
        assert input_seq_length + target_seq_length <= 20, "The maximum total sequence length is 20."
        
        try:
            filepath = tt.utils.data.download(MNIST_TEST_URL, 'tmp')
            data = np.float32(np.load(filepath))
        except:
            print 'Please set the correct path to the dataset. Might be caused by a download error.'
            sys.exit()

        # introduce channel dimension
        data = np.expand_dims(data, axis=4)
        # use value scale [0,1]
        data = data / 255.0  
        dataset_size = data.shape[0]
        
        super(MovingMNISTTestDataset, self).__init__(batch_size, data, dataset_size, (64, 64, 1),
                                                     input_seq_length, target_seq_length)
    @tt.utils.attr.override
    def get_batch(self):
        if self._row >= self._data.shape[0]:
            self.reset()
        
        batch_inputs = self._data[self._row:self._row+self._batch_size,
                                  0:self.input_seq_length,:,:,:]
        batch_targets = self._data[self._row:self._row+self._batch_size,
                                   self.input_seq_length:self.input_seq_length+self.target_seq_length,:,:,:]
        self._row = self._row + self._batch_size
        return batch_inputs, batch_targets