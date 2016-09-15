import os
import sys
from abc import ABCMeta

import h5py
import random
import numpy as np

import tensorflow as tf
import tensortools as tt
import base


# the file has do be downloaded manually
MSPAC_DOWNLOAD_URL = 'https://drive.google.com/file/d/0Byf787GZQ7KvV25xMWpWbV9LdUU'
MSPAC_FILENAME = "Ms_Pacman.zip"

FRAME_HEIGHT = 210
FRAME_WIDTH = 160
FRAME_CHANNELS = 3

# the y-pixel where the HUD at the bottom starts
HUD_Y = 172

# actually it's longer, but 100 is more than enough
MAX_SEQ_LENGTH = 100

SUBDIR_TRAIN = "Train"
SUBDIR_TEST = "Test"

# limit the retires in case we use no-change-skipping
# to ensure we do not end up in an endless-loop
MAX_SKIP_RETRIES = 2

    
class MsPacmanBaseDataset(base.AbstractDataset):
    """The MsPacman base dataset of the retro game classic.
       This dataset was used in "Adversarial Video Generation":
           https://github.com/dyelax/Adversarial_Video_Generation
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, subdir, index_range, data_dir, input_seq_length=10, target_seq_length=10,
                 crop_size=None, repetitions_per_epoche=256, skip_no_change=True,
                 random_flip=True):
        """Creates a MsPacman dataset instance.
        Parameters
        ----------
        subdir: str
            The subdirectory where to get the image data from for this dataset.
        index_range: tuple(start, end) or None
            The start and end index (inclusive!) of which sequences to take.
            Or None to use everything.
        data_dir: str
            The path where the data will be stored.
        input_seq_length: int, optional
            The input sequence length
        target_seq_length: int, optional
            The target sequence length
        crop_size: tuple(int) or None, optional
            The size (height, width) to randomly crop the images.
        repetitions_per_epoche: int, optioal
            Virtually increase the dataset size by a factor. We useually take only a small portion
            of the frame sequence that is up to ~1000 frames long. And in case of random cropping,
            we only take a small part of the image. That's why it is important to reuse these frame
            sequences multiple times, as we use a random part each time. If we would not, Testing would
            only require about two batches.
        skip_no_change: Boolean, optional
            Skip frame sequences where there is no change in the inputs
            at all.
        random_flip: Boolean, optional
            Whether we do random horizontal flip or not, as the game field is symmetric.
            In case cropping is active, we do not flip the frame in case the score-board
            at the bottom is visible.
        """
        assert input_seq_length + target_seq_length <= MAX_SEQ_LENGTH, "Maximum sequence length exceeded."
        
        # check or notify manual download
        filepath = os.path.join(data_dir, MSPAC_FILENAME)
        if not os.path.isfile(filepath):
            raise ValueError("{} not found.".format(filepath))
            
        dataset_path = tt.utils.data.extract(filepath, data_dir)
        self._data_dir = dataset_path
        
        train_dir = os.path.join(dataset_path, subdir)
        numbered_folder_paths = tt.utils.path.get_subdirpaths(train_dir)
        numbered_folder_paths.sort()
        
        if index_range is not None:
            # take only a specific part of the data
            numbered_folder_paths = numbered_folder_paths[index_range[0]:index_range[1]+1]
        
        # we save the file names, as well, to reduce string-ops at runtime
        # and names to not start with zero (0000.png)
        #data = [("path", files_count, [filenames])]
        data = []
        for nfp in numbered_folder_paths:
            filenames = tt.utils.path.get_filenames(nfp, "*.png", False)
            filenames.sort()
            data.append((nfp, len(filenames), filenames))
            
        self._data = data
        
        self._crop_size = crop_size
        self._repetitions_per_epoche = repetitions_per_epoche
        self._skip_no_change = skip_no_change
        self._random_flip = random_flip
            
        # virtually extend the dataset size by using these
        # long sequences multiple times, but different random parts
        self._true_dataset_size = len(data)
        dataset_size = self._true_dataset_size * repetitions_per_epoche
        
        self._input_seq_length = input_seq_length
        self._target_seq_length = target_seq_length
        
        super(MsPacmanBaseDataset, self).__init__(data_dir, dataset_size,
                                                  input_shape=[input_seq_length, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS],
                                                  target_shape=[target_seq_length, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS])
    @tt.utils.attr.override
    def get_batch(self, batch_size):
        total_seq_len = self._input_seq_length + self._target_seq_length
        
        #Every batch contains random sequence parts from randomly choses folders.
        seq_indices = np.random.choice(self._true_dataset_size, batch_size)
        
        if self._crop_size is None:
            input_batch_shape = [batch_size] + self.input_shape
            target_batch_shape = [batch_size] + self.target_shape
        else:
            input_batch_shape = [batch_size, self.input_shape[0],
                                 self._crop_size[0], self._crop_size[1], self.input_shape[3]]
            target_batch_shape = [batch_size, self.target_shape[0],
                                  self._crop_size[0], self._crop_size[1], self.target_shape[3]]
        
        batch_inputs = np.empty(input_batch_shape, dtype=np.uint8)
        batch_targets = np.empty(target_batch_shape, dtype=np.uint8)
        
        for batch in xrange(batch_size):
            current_seq = self._data[seq_indices[batch]]
            
            # select random frame index to start
            start_idx = random.randint(0, current_seq[1] - total_seq_len)
                
            do_flip = False
            if self._random_flip:
                # only flip when the hud is not visible
                if random.random() > 0.5:
                    do_flip = True
                    
            # pre-load input-frames fist. Because we do not need to process all
            # target frames in case we have to to a re-try due to "no-change" of pixels
            input_frames = []
            for fidx in xrange(start_idx, start_idx + self._input_seq_length):
                frame_path = os.path.join(current_seq[0], current_seq[2][fidx])
                input_frames.append(tt.utils.image.read(frame_path))
                
            
            for retry in range(MAX_SKIP_RETRIES):
                offset_y = offset_x = 0
                if self._crop_size is not None:
                    # do equal random crop
                    offset_x = random.randint(0, FRAME_WIDTH - self._crop_size[1])
                    offset_y = random.randint(0, FRAME_HEIGHT - self._crop_size[0])
                    
                    if offset_y + self._crop_size[0] < HUD_Y:
                        # undo flip in case the HUD is visible
                        do_flip = False
                
                # pre-load input-frames fist. Because we do not need to process all
                # target frames in case we have to to a re-try due to "no-change" of pixels
                for i in xrange(self._input_seq_length):
                    frame = input_frames[i]
                    
                    if self._crop_size is not None:
                        # crop image
                        frame = frame[offset_y:(offset_y + self._crop_size[0]),
                                      offset_x:(offset_x + self._crop_size[1]),:]
                        
                    # add to batch
                    batch_inputs[batch, i] = frame[:,::-1,:] if do_flip else frame
                    
                # do lazy check? (1st, 5th, last?)
                if self._skip_no_change and retry != MAX_SKIP_RETRIES:
                    # check for at least 1-pixel changel in the input-frames
                    change = False
                    for y in xrange(self._input_seq_length - 1):
                        if not np.array_equal(batch_inputs[batch, y], batch_inputs[batch, y + 1]):
                            change = True
                            break
                    if not change:
                        # do a retry because all frames are identical
                        continue
                    
                # we reach this, when the input sequence was valid and we can now
                # start to process the targets as well
                for i, fidx in enumerate(xrange(start_idx + self._input_seq_length, start_idx + total_seq_len)):
                    frame_path = os.path.join(current_seq[0], current_seq[2][fidx])
                    frame = tt.utils.image.read(frame_path)
                    
                    if self._crop_size is not None:
                        # crop image
                        frame = frame[offset_y:(offset_y + self._crop_size[0]),
                                      offset_x:(offset_x + self._crop_size[1]),:]
                        
                    # add to batch
                    batch_targets[batch, i] = frame[:,::-1,:] if do_flip else frame
                
                # break to stop retrying
                break
        
        # to scale [0, 1] as type float32
        batch_inputs = batch_inputs / np.float32(255)
        batch_targets = batch_targets / np.float32(255)
        
        return batch_inputs, batch_targets
    
    @tt.utils.attr.override
    def reset(self):
        pass
    
    
class MsPacmanTrainDataset(MsPacmanBaseDataset):
    """The MsPacman training dataset of the retro game classic.
       This dataset was used in "Adversarial Video Generation":
           https://github.com/dyelax/Adversarial_Video_Generation
    """
    def __init__(self, data_dir, input_seq_length=10, target_seq_length=10,
                 crop_size=None, repetitions_per_epoche=256, skip_no_change=True,
                 random_flip=True):
        """Creates a MsPacman dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_seq_length: int, optional
            The input sequence length
        target_seq_length: int, optional
            The target sequence length
        crop_size: tuple(int) or None, optional
            The size (height, width) to randomly crop the images.
        repetitions_per_epoche: int, optioal
            Virtually increase the dataset size by a factor. We useually take only a small portion
            of the frame sequence that is up to ~1000 frames long. And in case of random cropping,
            we only take a small part of the image. That's why it is important to reuse these frame
            sequences multiple times, as we use a random part each time. If we would not, Testing would
            only require about two batches.
        skip_no_change: Boolean, optional
            Skip frame sequences where there is no change in the inputs
            at all.
        random_flip: Boolean, optional
            Whether we do random horizontal flip or not, as the game field is symmetric.
            In case cropping is active, we do not flip the frame in case the score-board
            at the bottom is visible.
        """
        super(MsPacmanTrainDataset, self).__init__(SUBDIR_TRAIN, (0, 465), data_dir, input_seq_length, target_seq_length,
                                                   crop_size, repetitions_per_epoche, skip_no_change, random_flip)
    
    
class MsPacmanValidDataset(MsPacmanBaseDataset):
    """The MsPacman validation dataset of the retro game classic.
       This dataset was used in "Adversarial Video Generation":
           https://github.com/dyelax/Adversarial_Video_Generation
    """
    def __init__(self, data_dir, input_seq_length=10, target_seq_length=10,
                 crop_size=None, repetitions_per_epoche=256, skip_no_change=True,
                 random_flip=True):
        """Creates a MsPacman dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_seq_length: int, optional
            The input sequence length
        target_seq_length: int, optional
            The target sequence length
        crop_size: tuple(int) or None, optional
            The size (height, width) to randomly crop the images.
        repetitions_per_epoche: int, optioal
            Virtually increase the dataset size by a factor. We useually take only a small portion
            of the frame sequence that is up to ~1000 frames long. And in case of random cropping,
            we only take a small part of the image. That's why it is important to reuse these frame
            sequences multiple times, as we use a random part each time. If we would not, Testing would
            only require about two batches.
        skip_no_change: Boolean, optional
            Skip frame sequences where there is no change in the inputs
            at all.
        random_flip: Boolean, optional
            Whether we do random horizontal flip or not, as the game field is symmetric.
            In case cropping is active, we do not flip the frame in case the score-board
            at the bottom is visible.
        """
        super(MsPacmanValidDataset, self).__init__(SUBDIR_TRAIN, (466 ,516), data_dir, input_seq_length, target_seq_length,
                                                   crop_size, repetitions_per_epoche, skip_no_change, random_flip)
        

class MsPacmanTestDataset(MsPacmanBaseDataset):
    """The MsPacman test dataset of the retro game classic.
       This dataset was used in "Adversarial Video Generation":
           https://github.com/dyelax/Adversarial_Video_Generation
    """
    def __init__(self, data_dir, input_seq_length=10, target_seq_length=10,
                 crop_size=None, repetitions_per_epoche=256, skip_no_change=True,
                 random_flip=True):
        """Creates a MsPacman dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_seq_length: int, optional
            The input sequence length
        target_seq_length: int, optional
            The target sequence length
        crop_size: tuple(int) or None, optional
            The size (height, width) to randomly crop the images.
        repetitions_per_epoche: int, optioal
            Virtually increase the dataset size by a factor. We useually take only a small portion
            of the frame sequence that is up to ~1000 frames long. And in case of random cropping,
            we only take a small part of the image. That's why it is important to reuse these frame
            sequences multiple times, as we use a random part each time. If we would not, Testing would
            only require about two batches.
        skip_no_change: Boolean, optional
            Skip frame sequences where there is no change in the inputs
            at all.
        random_flip: Boolean, optional
            Whether we do random horizontal flip or not, as the game field is symmetric.
            In case cropping is active, we do not flip the frame in case the score-board
            at the bottom is visible.
        """
        super(MsPacmanTestDataset, self).__init__(SUBDIR_TEST, None, data_dir, input_seq_length, target_seq_length,
                                                  crop_size, repetitions_per_epoche, skip_no_change, random_flip)