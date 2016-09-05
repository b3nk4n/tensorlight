import sys
import os
import random

import numpy as np
import tensorflow as tf
import tensortools as tt
import base


UCF11_URL = 'http://crcv.ucf.edu/data/UCF11_updated_mpg.rar'

FRAME_HEIGHT = 240
FRAME_WIDTH = 320
FRAME_CHANNELS = 3


def _serialize_frame_sequences(dataset_path, image_size, serialized_sequence_length):
    sequence_files = tt.utils.path.get_filenames(dataset_path, '*.seq')
    
    if len(sequence_files) > 0:
        # Test if image_size has changed
        example = np.fromfile(sequence_files[0], np.uint8)
        old_ndim = np.prod(example.shape)
        new_ndim = np.prod(image_size) * serialized_sequence_length
        if old_ndim == new_ndim:
            # Reuse old serialized files
            dataset_size = len(sequence_files)
            print("Found {} serialized frame sequences. Skipping serialization.".format(dataset_size))
            return dataset_size
        else:
            print("Change in image properties detected. Deleting previous serialized *.seq files...")
            # remove old serialized files
            for sfile in sequence_files:
                os.remove(sfile)            

    video_filenames = tt.utils.path.get_filenames(dataset_path, '*.mpg')
    frame_scale_factor = image_size[0] / float(FRAME_HEIGHT)
    
    print("Serializing frame sequences...")
    success_counter = 0
    bounds_counter = 0
    short_counter = 0
    progress = tt.utils.ui.ProgressBar(len(video_filenames))
    for i, video_filename in enumerate(video_filenames):
        with tt.utils.video.VideoReader(video_filename) as vr:
            # until we reach the end of the video
            clip_id = 0
            while True:
                frames = []
                for f in xrange(serialized_sequence_length):
                    frame = vr.next_frame(frame_scale_factor)
                    
                    if frame is None:
                        break
                        
                    # ensure frame is not too large
                    h, w, c = np.shape(frame)
                    if h > image_size[0] or w > image_size[1]:
                        frame = frame[:image_size[0], :image_size[1], :]
                    if not h < image_size[0] and not w < image_size[1]:
                        frame = np.reshape(frame, [image_size[0], image_size[1], -1])
                        if image_size[2] == 1:
                            frame = tt.utils.image.to_grayscale(frame)
                        frames.append(frame)
                    else:
                        # clip has wrong bounds
                        bounds_counter += 1
                        break

                if len(frames) == serialized_sequence_length:
                    seq_filepath = "{}-{}.seq".format(os.path.splitext(video_filename)[0], clip_id)
                    tt.utils.image.write_as_binary(seq_filepath, np.asarray(frames))
                    success_counter += 1
                    clip_id += 1
                else:
                    if clip_id == 0:
                        # clip end reached in first run (video was not used at all!)
                        short_counter += 1
                    break
        progress.update(i+1)
    print("Successfully extracted {} frame sequences. Too short: {}, Too small bounds: {}" \
          .format(success_counter, short_counter, bounds_counter))
    return success_counter


class UCF11TrainDataset(base.AbstractQueueDataset):
    """UCF-11 sports dataset that creates a bunch of binary frame sequences
       and uses a file queue for multi-threaded input reading.
       
       References: http://crcv.ucf.edu/data/UCF_YouTube_Action.php
    """
    def __init__(self, input_seq_length=5, target_seq_length=5,
                 image_size=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
                 min_examples_in_queue=512, queue_capacitiy=1024, num_threads=8,
                 serialized_sequence_length=30, do_distortion=True):
        """Creates a training dataset instance that uses a queue.
        Parameters
        ----------
        dataset_size: int, optional
            The dataset site.
        input_seq_length: int, optional
            The length of the input sequence.
        target_seq_length: length
            The length of the target sequence.
        image_size: list(int) of shape [h, w, c]
            The image size, how the data with default scale [240, 320, 3]
            should be scaled to.
        min_examples_in_queue: int, optional
            The minimum examples that have to be in the queue.
            A higher value ensures a good mix.
        queue_capacitiy: int, optional
            The maximum capacity of the input queue.
        num_threads: int, optional
            The number of threads to generate the inputs.
        serialized_sequence_length: int, optional
            The sequence length of each serialized file.
        do_distortion: Boolean, optional
            Whether image distortion should be performed or not.
            Can have a very bad influence on performance.
        """
        self._serialized_sequence_length = serialized_sequence_length
        self._do_distortion = do_distortion
        
        try:
            rar_path = tt.utils.data.download(UCF11_URL, 'tmp')
            dataset_path = tt.utils.data.extract(rar_path, 'tmp')
            self._data_dir = dataset_path
           
        except:
            print 'Please set the correct path to UCF11 dataset. Might be caused by a download error.'
            sys.exit()
        
        dataset_size = _serialize_frame_sequences(dataset_path,
                                                  image_size,
                                                  serialized_sequence_length)
        super(UCF11TrainDataset, self).__init__(dataset_size, [input_seq_length, image_size[0], image_size[1], image_size[2]],
                                                [target_seq_length, image_size[0], image_size[1], image_size[2]],
                                                min_examples_in_queue, queue_capacitiy, num_threads)
    
    def _read_record(self, filename_queue):
        
        class FrameSeqRecord(object):
            pass
        
        record = FrameSeqRecord()
        record.height = self.input_shape[1]
        record.width = self.input_shape[2]
        record.depth = self.input_shape[3]
        
        input_seq_length = self.input_shape[0]
        target_seq_length = self.target_shape[0]
        total_seq_length = input_seq_length + target_seq_length

        frame_bytes = record.height * record.width * record.depth
        record_bytes = frame_bytes * (total_seq_length)
        total_file_bytes = frame_bytes * self._serialized_sequence_length

        with tf.name_scope('read_record'):
            reader = tf.FixedLengthRecordReader(total_file_bytes)

            record.key, value = reader.read(filename_queue)
            decoded_record_bytes = tf.decode_raw(value, tf.uint8)

            record.data = decoded_record_bytes[0:input_seq_length]

            decoded_record_bytes = tf.reshape(decoded_record_bytes,
                                              [self._serialized_sequence_length, record.height, record.width, record.depth])

            # calculcate tensors [start, 0, 0, 0]
            rnd_start_index = tf.to_int32(tf.random_uniform([1], 0, self._serialized_sequence_length - (total_seq_length), 
                                                            tf.int32))
            seq_start_offset = tf.SparseTensor(indices=[[0]], values=rnd_start_index, shape=[4])
            sequence_start = tf.sparse_tensor_to_dense(seq_start_offset)

            # take first frames as input
            record.data = tf.cast(tf.slice(decoded_record_bytes, sequence_start,
                                           [total_seq_length, record.height, record.width, record.depth]),
                                  tf.float32)
            return record

    @tt.utils.attr.override
    def get_batch(self, batch_size):
        # Generate a batch of sequences and labels by building up a queue of examples.
        seq_filenames = tt.utils.path.get_filenames(self._data_dir, '*.seq')
        with tf.name_scope('preprocessing'):
            filename_queue = tf.train.string_input_producer(seq_filenames)
            seq_record = self._read_record(filename_queue)  

            # convert to float of scale [0.0, 1.0]
            reshaped_seq = tf.cast(seq_record.data, tf.float32)
            reshaped_seq = reshaped_seq / 255
    
            input_seq_length = self.input_shape[0]
            target_seq_length = self.target_shape[0]
            total_seq_length = input_seq_length + target_seq_length
            if self._do_distortion:
                with tf.name_scope('distort_inputs'):
                    images_to_distort = []
                    for i in xrange(total_seq_length):
                        images_to_distort.append(reshaped_seq[i,:,:,:])

                    distorted_images = tt.image.equal_random_distortion(images_to_distort)
                    sequence_inputs = tf.pack(distorted_images[0:input_seq_length], axis=0)
                    sequence_targets = tf.pack(distorted_images[input_seq_length:], axis=0)
            else:
                sequence_inputs = reshaped_seq[0:input_seq_length,:,:,:]
                sequence_targets = reshaped_seq[input_seq_length:,:,:,:]

        return tt.inputs.generate_batch(sequence_inputs, sequence_targets,
                                        batch_size,
                                        self._min_examples_in_queue, self._queue_capacitiy,
                                        shuffle=True, num_threads=self._num_threads)

    @property
    def serialized_sequence_length(self):
        """Gets the serialized sequence length"""
        return self._serialized_sequence_length
    
    @property
    def do_distortion(self):
        """Gets whether distorion is activated."""
        return self._do_distortion
    
    
class UCF11ValidDataset(base.AbstractDataset):    
    """UCF-11 sports dataset that creates a bunch of binary frame sequences.
       This is a pretty bad validation set implementation, as it uses the same
       examples as the training set, with only these differences:
           - It uses no no random contrast, brightness.
           - All frames are flipped horizontaly, just to be different from
             a training set without distortion.
       
       References: http://crcv.ucf.edu/data/UCF_YouTube_Action.php
    """
    def __init__(self, input_seq_length=5, target_seq_length=5,
                 image_size=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
                 serialized_sequence_length=30, do_distortion=True):
        """Creates a validation dataset instance.
        Parameters
        ----------
        dataset_size: int, optional
            The dataset site.
        input_seq_length: int, optional
            The length of the input sequence.
        target_seq_length: length
            The length of the target sequence.
        image_size: list(int) of shape [h, w, c]
            The image size, how the data with default scale [240, 320, 3]
            should be scaled to.
        serialized_sequence_length: int, optional
            The sequence length of each serialized file.
        do_distortion: Boolean, optional
            Whether image distortion should be performed or not.
            Can have a very bad influence on performance.
        """
        self._serialized_sequence_length = serialized_sequence_length
        self._do_distortion = do_distortion
        
        try:
            rar_path = tt.utils.data.download(UCF11_URL, 'tmp')
            dataset_path = tt.utils.data.extract(rar_path, 'tmp')
            self._data_dir = dataset_path
        except:
            print 'Please set the correct path to UCF11 dataset. Might be caused by a download error.'
            sys.exit()
        
        dataset_size = _serialize_frame_sequences(dataset_path,
                                                  image_size,
                                                  serialized_sequence_length)
        
        self._file_name_list = tt.utils.path.get_filenames(self._data_dir, '*.seq')
        self._indices = range(dataset_size)
        self._row = 0
        
        super(UCF11ValidDataset, self).__init__(dataset_size, [input_seq_length, image_size[0], image_size[1], image_size[2]],
                                                [target_seq_length, image_size[0], image_size[1], image_size[2]])

    @tt.utils.attr.override
    def get_batch(self, batch_size):
        if self._row + batch_size >= self.size:
            self.reset()
        start = self._row
        end = start + batch_size
        ind_range = self._indices[start:end]
        self._row += batch_size
        
        # get next filenames
        file_names = [self._file_name_list[i] for i in ind_range]
        
        # evaluate if we to random flip
        do_flip = False
        if self.do_distortion:
            if random.random() > 0.5:
                do_flip = True
        
        # load serialized sequences
        seq_input_list = []
        seq_target_list = []
        for f in file_names:
            current = tt.utils.image.read_as_binary(f, dtype=np.uint8)
            current = np.reshape(current, [self.serialized_sequence_length] + self.input_shape[1:])
            
            if do_flip:
                #current = np.flip(current, axis=-2) # only available in numpy 1.1.12 dev0
                current = current[:,:,::-1,:] # horizontal flip
            
            seq_input_list.append(current[0:self.input_shape[0]])
            seq_target_list.append(current[self.input_shape[0]:self.input_shape[0]+self.target_shape[0]])
        
        input_sequence = np.stack(seq_input_list)
        target_sequence = np.stack(seq_target_list)
        
        # convert to float of scale [0.0, 1.0]
        inputs = input_sequence.astype(np.float32) / 255
        targets = target_sequence.astype(np.float32) / 255
        
        return inputs, targets
    
    @tt.utils.attr.override
    def reset(self):
        self._row = 0
        np.random.shuffle(self._indices)
        
    @property
    def serialized_sequence_length(self):
        """Gets the serialized sequence length"""
        return self._serialized_sequence_length
    
    @property
    def do_distortion(self):
        """Gets whether distorion is activated."""
        return self._do_distortion