import sys

import h5py
import numpy as np
import tensortools as tt


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



class MovingMNISTTrainDataset(object):
    """Moving MNIST dataset that creates data on the fly."""
    def __init__(self, batch_size, num_frames, image_size=(64, 64), num_digits=2, step_length=0.1):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        """
        self._seq_length = num_frames
        self._batch_size = batch_size
        self._image_size = image_size
        self._num_digits = num_digits
        self._step_length = step_length
        self._dataset_size = sys.maxint
        self._digit_size = 28
        self._frame_size = image_size[0] * image_size[1] # is this really needed?

        try:
            filepath = tt.utils.data.download(MNIST_URL, 'tmp')
            f = h5py.File(filepath)
        except:
            print 'Please set the correct path to MNIST dataset. Might be caused by a download error.'
            sys.exit()

        self._data = f['train'].value.reshape(-1, 28, 28)
        f.close()
        self._indices = np.arange(self._data.shape[0])
        self._row = 0
        self.reset()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dims(self): # TODO: is this used somewhere? If not, delete!
        return self._frame_size
    
    @property
    def inputs_shape(self): # TODO: self._seq_length // 2: split into input_seq_length and targets_seq_length
        return [self._seq_length + 1 // 2, self._image_size[0], self._image_size[1], 1]
    
    @property
    def targets_shape(self): # TODO: self._seq_length // 2: split into input_seq_length and targets_seq_length
        return [self._seq_length - 1 // 2, self._image_size[0], self._image_size[1], 1]

    @property
    def dataset_size(self):
        return self._dataset_size

    @property
    def seq_length(self):
        return self._seq_length
    
    def reset(self):
        self._row = 0
        np.random.shuffle(self._indices)
        pass

    def get_batch(self):
        start_y, start_x = _get_random_trajectory(self._batch_size * self._num_digits,
                                                  self._seq_length, 
                                                  self._image_size, 
                                                  self._digit_size, 
                                                  self._step_length)
    
        # minibatch data
        data = np.zeros((self._batch_size, self._seq_length, self._image_size[0], self._image_size[1], 1),
                        dtype=np.float32)
    
        for j in xrange(self._batch_size):
            for n in xrange(self._num_digits):
       
                # get random digit from dataset
                ind = self._indices[self._row]
                self._row += 1
                if self._row == self._data.shape[0]:
                    self.reset()
                digit_image = self._data[ind, :, :]
        
                # generate video
                for i in xrange(self._seq_length):
                    top    = start_y[i, j * self._num_digits + n]
                    left   = start_x[i, j * self._num_digits + n]
                    bottom = top  + self._digit_size
                    right  = left + self._digit_size
                    # set data and use maximum for overlap
                    data[j, i, top:bottom, left:right, 0] = np.maximum(data[j, i, top:bottom, left:right, 0],
                                                                       digit_image)
    
        return data
    
    
    
class MovingMNISTValidDataset(object):
    """Moving MNIST dataset for validation."""
    def __init__(self, batch_size, num_frames, image_size=(64, 64), num_digits=2, step_length=0.1):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        """
        self._seq_length = num_frames
        self._batch_size = batch_size
        self._image_size = image_size
        self._num_digits = num_digits
        self._step_length = step_length
        self._dataset_size = 10000
        self._digit_size = 28
        self._frame_size = image_size[0] * image_size[1]

        try:
            filepath = tt.utils.data.download(MNIST_URL, 'tmp')
            f = h5py.File(filepath)
        except:
            print 'Please set the correct path to MNIST dataset. Might be caused by a download error.'
            sys.exit()

        self._data = f['validation'].value.reshape(-1, 28, 28)
        f.close()
        self._indices = np.arange(self._data.shape[0])
        self._row = 0
        self.reset()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dims(self):
        return self._frame_size

    @property
    def dataset_size(self):
        return self._dataset_size

    @property
    def seq_length(self):
        return self._seq_length
    
    def reset(self):
        self._row = 0
        np.random.shuffle(self._indices)
        pass

    def get_batch(self):
        start_y, start_x = _get_random_trajectory(self._batch_size * self._num_digits,
                                                  self._seq_length, 
                                                  self._image_size, 
                                                  self._digit_size, 
                                                  self._step_length)
    
        # minibatch data
        data = np.zeros((self._batch_size, self._seq_length, self._image_size[0], self._image_size[1], 1), 
                        dtype=np.float32)
    
        for j in xrange(self._batch_size):
            for n in xrange(self._num_digits):
       
                # get random digit from dataset
                ind = self._indices[self._row]
                self._row += 1
                if self._row == self._data.shape[0]:
                    self.reset()
                digit_image = self._data[ind, :, :]
        
                # generate video
                for i in xrange(self._seq_length):
                    top    = start_y[i, j * self._num_digits + n]
                    left   = start_x[i, j * self._num_digits + n]
                    bottom = top  + self._digit_size
                    right  = left + self._digit_size
                    # set data and use maximum for overlap
                    data[j, i, top:bottom, left:right, 0] = np.maximum(data[j, i, top:bottom, left:right, 0],
                                                                       digit_image)
    
        return data

    
    
# video patches loaded from some file
class MovingMNISTTestDataset(object):
    def __init__(self, batch_size, num_frames):
        self._seq_length = num_frames
        self._batch_size = batch_size
        self._image_size = (64, 64)
        self._frame_size = self._image_size[0] * self._image_size[1]

        try:
            filepath = tt.utils.data.download(MNIST_TEST_URL, 'tmp')
            self._data = np.float32(np.load(filepath))
            # introduce channel dimension
            self._data = np.expand_dims(self._data, axis=4)
            # use value scale [0,1]
            self._data = self._data / 255.0  
        except:
            print 'Please set the correct path to the dataset. Might be caused by a download error.'
            sys.exit()

        self._dataset_size = self._data.shape[0]
        self._row = 0
        self.reset()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dims(self):
        return self._frame_size

    @property
    def dataset_size(self):
        return self._dataset_size

    @property
    def seq_length(self):
        return self._seq_length

    def reset(self):
        self._row = 0
        pass

    def get_batch(self):
        minibatch = self._data[self._row:self._row+self._batch_size]    
        self._row = self._row + self._batch_size
    
        if self._row == self._data.shape[0]:
            self.reset()
    
        return minibatch