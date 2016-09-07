import sys
import os
import random

import numpy as np
import tensorflow as tf
import tensortools as tt
import base


UCF101_URL = 'http://crcv.ucf.edu/data/UCF101/UCF101.rar'
UCF101_SPLITS_URL = 'http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip'
UCF101_TRAINLIST = "trainlist03.txt"
UCF101_TESTLIST = "testlist03.txt"

FRAME_HEIGHT = 240
FRAME_WIDTH = 320
FRAME_CHANNELS = 3


def _read_train_splits(dir_path):
    """Reads the filepaths of the train split."""
    train_files = []
    
    filepath = os.path.join(dir_path, UCF101_TRAINLIST)
    with open(filepath) as f:
        for line in f:
            train_files.append(line.split()[0])
    return train_files


def _read_test_splits(dir_path):
    """Reads the filepaths of the valid/test splits."""
    test_files = []
    valid_files = []
    
    filepath = os.path.join(dir_path, UCF101_TESTLIST)
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i % 3 == 0:
                valid_files.append(line.split()[0])
            else:    
                test_files.append(line.split()[0])
            
    return valid_files, test_files


def _serialize_frame_sequences(dataset_path, file_list, image_size, serialized_sequence_length):
    seq_file_list = []
    for f in file_list:
        # change extension to *.seq
        seq_file_list.append("{}.seq".format(os.path.splitext(f)[0]))
    
    # Test if image_size has changed
    first_seq_file = os.path.join(dataset_path, seq_file_list[0])
    if os.path.isfile(first_seq_file):
        example = np.fromfile(first_seq_file, np.uint8)
        print("dims", old_ndim, new_ndim)
        old_ndim = np.prod(example.shape)
        new_ndim = np.prod(image_size) * serialized_sequence_length
        if old_ndim == new_ndim:
            # Reuse old serialized files
            dataset_size = len(seq_file_list)
            print("Found {} serialized frame sequences. Skipping serialization.".format(dataset_size))
            return dataset_size
        else:
            print("Change in image properties detected. Deleting previous serialized *.seq files...")
            # remove old serialized files
            for sfile in seq_file_list:
                file_to_delete = os.path.join(dataset_path, sfile)
                os.remove(file_to_delete)            

    frame_scale_factor = image_size[0] / float(FRAME_HEIGHT)
    
    print("Serializing frame sequences...")
    success_counter = 0
    bounds_counter = 0
    short_counter = 0
    progress = tt.utils.ui.ProgressBar(len(file_list))
    for i, filename in enumerate(file_list):
        video_filename = os.path.join(dataset_path, filename)
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


class UCF101TrainDataset(base.AbstractQueueDataset):
    """UCF-101 dataset that creates a bunch of binary frame sequences
       and uses a file queue for multi-threaded input reading.
       
       References: http://crcv.ucf.edu/data/UCF101.php
    """
    def __init__(self, input_seq_length=5, target_seq_length=5,
                 image_size=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
                 min_examples_in_queue=1024, queue_capacitiy=2048, num_threads=8,
                 serialized_sequence_length=30, do_distortion=True, crop_size=None):
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
        crop_size: tuple(int) or None, optional
            The size (height, width) to randomly crop the images.
        """
        if crop_size is not None:
            assert image_size[0] > crop_size[0] and image_size[1] > crop_size[1], \
                "Image size has to be larger than the crop size."
        
        
        self._serialized_sequence_length = serialized_sequence_length
        self._do_distortion = do_distortion
        self._crop_size = crop_size
        self._data_img_size = image_size
        
        rar_path = tt.utils.data.download(UCF101_URL, 'tmp')

        dataset_path = tt.utils.data.extract(rar_path, 'tmp', unpacked_name='UCF-101')
        self._data_dir = dataset_path
            
        zip_path = tt.utils.data.download(UCF101_SPLITS_URL, 'tmp')
        splits_path = tt.utils.data.extract(zip_path, 'tmp', unpacked_name='ucfTrainTestlist')
            
        train_files = _read_train_splits(splits_path)
        dataset_size = _serialize_frame_sequences(dataset_path, train_files,
                                                  image_size,
                                                  serialized_sequence_length)
        
        if crop_size is None:
            input_shape = [input_seq_length, image_size[0], image_size[1], image_size[2]]
            target_shape = [target_seq_length, image_size[0], image_size[1], image_size[2]]
        else:
            input_shape = [input_seq_length, crop_size[0], crop_size[1], image_size[2]]
            target_shape = [target_seq_length, crop_size[0], crop_size[1], image_size[2]]
        
        super(UCF101TrainDataset, self).__init__(dataset_size, input_shape, target_shape,
                                                min_examples_in_queue, queue_capacitiy, num_threads)
    
    def _read_record(self, filename_queue):
        
        class FrameSeqRecord(object):
            pass
        
        record = FrameSeqRecord()
        record.height = self._data_img_size[0]
        record.width = self._data_img_size[1]
        record.depth = self._data_img_size[2]
        
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
            seq_data = tf.cast(seq_record.data, tf.float32)
            seq_data = seq_data / 255
    
            input_seq_length = self.input_shape[0]
            target_seq_length = self.target_shape[0]
            total_seq_length = input_seq_length + target_seq_length
            
            if self._crop_size is not None:
                with tf.name_scope('random_crop'):
                    seq_data = tf.random_crop(seq_data,
                                              [total_seq_length,
                                               self._crop_size[0],
                                               self._crop_size[1],
                                               self.input_shape[3]])
            if self._do_distortion:
                with tf.name_scope('distortion'):
                    images_to_distort = tf.unpack(seq_data)

                    distorted_images = tt.image.equal_random_distortion(images_to_distort)
                    sequence_inputs = tf.pack(distorted_images[0:input_seq_length], axis=0)
                    sequence_targets = tf.pack(distorted_images[input_seq_length:], axis=0)
            else:
                sequence_inputs = seq_data[0:input_seq_length,:,:,:]
                sequence_targets = seq_data[input_seq_length:,:,:,:]

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