import sys

import h5py
import numpy as np
import tensortools as tt


# we use the same MNIST dataset as University of Toronto in its 'Unsupervised Learning with LSTMS'
# paper to make sure we can compare our results with theirs.
MNIST_URL = 'http://www.cs.toronto.edu/~emansim/datasets/mnist.h5'


class MovingMNISTDataset(object):
    """Moving MNIST dataset that creates data on the fly."""
    def __init__(self, batch_size, num_frames, image_size=64, num_digits=2, step_length=0.1):
        """Creates a dataset instance.
        Reference: Based on Srivastava et al.
                   http://www.cs.toronto.edu/~nitish/unsupervised_video/
        Parameters
        ----------
        ... TODO: describe parameters of this class.
        """
        self.seq_length_ = num_frames
        self.batch_size_ = batch_size
        self.image_size_ = image_size
        self.num_digits_ = num_digits
        self.step_length_ = step_length
        self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
        self.digit_size_ = 28
        self.frame_size_ = self.image_size_ ** 2

        try:
            filepath = tt.utils.data.download(MNIST_URL, 'tmp')
            f = h5py.File(filepath)
        except:
            print 'Please set the correct path to MNIST dataset'
            sys.exit()

        self.data_ = f['train'].value.reshape(-1, 28, 28)
        f.close()
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        np.random.shuffle(self.indices_)

    @property
    def batch_size(self):
        return self.batch_size_

    @property
    def dims(self):
        return self.frame_size_

    @property
    def dataset_size(self):
        return self.dataset_size_

    @property
    def seq_length(self):
        return self.seq_length_

    def _get_random_trajectory(self, batch_size):
        length = self.seq_length_
        canvas_size = self.image_size_ - self.digit_size_

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
            y += v_y * self.step_length_
            x += v_x * self.step_length_

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
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def _overlap(self, a, b):
        """ Put b on top of a."""
        return np.maximum(a, b)

    def get_batch(self, verbose=False):
        start_y, start_x = self._get_random_trajectory(self.batch_size_ * self.num_digits_)
    
        # minibatch data
        data = np.zeros((self.batch_size_, self.seq_length_, self.image_size_, self.image_size_, 1), dtype=np.float32)
    
        for j in xrange(self.batch_size_):
            for n in xrange(self.num_digits_):
       
                # get random digit from dataset
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self.row_ = 0
                    np.random.shuffle(self.indices_)
                digit_image = self.data_[ind, :, :]
        
                # generate video
                for i in xrange(self.seq_length_):
                    top    = start_y[i, j * self.num_digits_ + n]
                    left   = start_x[i, j * self.num_digits_ + n]
                    bottom = top  + self.digit_size_
                    right  = left + self.digit_size_
                    data[j, i, top:bottom, left:right, 0] = self._overlap(data[j, i, top:bottom, left:right, 0], digit_image)
    
        return data
