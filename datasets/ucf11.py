import sys
import os
import random

import numpy as np
import tensorflow as tf

import tensorlight as light
import base


UCF11_URL = 'http://crcv.ucf.edu/data/UCF11_updated_mpg.rar'

SUBDIR_SHARED = '_shared'

FRAME_HEIGHT = 240
FRAME_WIDTH = 320
FRAME_CHANNELS = 3


class UCF11TrainDataset(base.AbstractQueueDataset):
    """UCF-11 sports dataset that creates a bunch of binary frame sequences
       and uses a file queue for multi-threaded input reading.
       
       References: http://crcv.ucf.edu/data/UCF_YouTube_Action.php
    """
    def __init__(self, data_dir, input_seq_length=5, target_seq_length=5,
                 image_scale_factor=1.0, gray_scale=False,
                 min_examples_in_queue=1024, queue_capacitiy=2048, num_threads=16,
                 serialized_sequence_length=30, do_distortion=True, crop_size=None):
        """Creates a training dataset instance that uses a queue.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_seq_length: int, optional
            The length of the input sequence.
        target_seq_length: length
            The length of the target sequence.
        image_scale_factor: float in range (0.0, 1.0], optional
            The image scale size, how the data should be scaled to.
        gray_scale: Boolean, optional
            Whether we scale the image to gray or not.
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
        image_size = [int(FRAME_HEIGHT * image_scale_factor),
                      int(FRAME_WIDTH * image_scale_factor),
                      1 if gray_scale else FRAME_CHANNELS]
        
        if crop_size is not None:
            assert image_size[0] > crop_size[0] and image_size[1] > crop_size[1], \
                "Image size has to be larger than the crop size."
        
        self._serialized_sequence_length = serialized_sequence_length
        self._do_distortion = do_distortion
        self._crop_size = crop_size
        self._data_img_size = image_size
        
        # download and extract data
        rar_path = light.utils.data.download(UCF11_URL, data_dir)
        dataset_path = light.utils.data.extract(rar_path, data_dir)
        self._data_dir = dataset_path
        
        # generate frame sequences.
        video_filenames = light.utils.path.get_filenames(dataset_path, '*.mpg')
        dataset_size, seq_files = light.utils.data.preprocess_videos(dataset_path, SUBDIR_SHARED,
                                                          video_filenames,
                                                          [FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS],
                                                          serialized_sequence_length,
                                                          gray_scale, image_scale_factor)
        self._file_name_list = seq_files
        
        if crop_size is None:
            input_shape = [input_seq_length, image_size[0], image_size[1], image_size[2]]
            target_shape = [target_seq_length, image_size[0], image_size[1], image_size[2]]
        else:
            input_shape = [input_seq_length, crop_size[0], crop_size[1], image_size[2]]
            target_shape = [target_seq_length, crop_size[0], crop_size[1], image_size[2]]
        
        super(UCF11TrainDataset, self).__init__(data_dir, dataset_size, input_shape, target_shape,
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

            decoded_record_bytes = tf.reshape(decoded_record_bytes,
                                              [self._serialized_sequence_length, record.height, record.width, record.depth])

            # calculcate tensors [start, 0, 0, 0]
            rnd_start_index = tf.to_int32(tf.random_uniform([1], 0, self._serialized_sequence_length - (total_seq_length), 
                                                            tf.int32))
            seq_start_offset = tf.SparseTensor(indices=[[0]], values=rnd_start_index, shape=[4])
            sequence_start = tf.sparse_tensor_to_dense(seq_start_offset)

            # take a random slice of frames as input
            record.data = tf.slice(decoded_record_bytes, sequence_start,
                                   [total_seq_length, record.height, record.width, record.depth])
            return record

    @light.utils.attr.override
    def get_batch(self, batch_size):
        # Generate a batch of sequences and labels by building up a queue of examples.
        with tf.name_scope('preprocessing'):
            filename_queue = tf.train.string_input_producer(self._file_name_list,
                                                            capacity=128)
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
                    images_to_distort = []
                    for i in xrange(total_seq_length):
                        images_to_distort.append(seq_data[i,:,:,:])

                    distorted_images = light.image.equal_random_distortion(images_to_distort)
                    sequence_inputs = tf.pack(distorted_images[0:input_seq_length], axis=0)
                    sequence_targets = tf.pack(distorted_images[input_seq_length:], axis=0)
            else:
                sequence_inputs = seq_data[0:input_seq_length,:,:,:]
                sequence_targets = seq_data[input_seq_length:,:,:,:]

        #return sequence_inputs, sequence_targets     
        return light.inputs.generate_batch(sequence_inputs, sequence_targets,
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
    def __init__(self, data_dir, input_seq_length=5, target_seq_length=5,
                 image_scale_factor=1.0, gray_scale=False,
                 serialized_sequence_length=30, do_distortion=True, crop_size=None):
        """Creates a validation dataset instance.
        Parameters
        ----------
        data_dir: str
            The path where the data will be stored.
        input_seq_length: int, optional
            The length of the input sequence.
        target_seq_length: length
            The length of the target sequence.
        image_scale_factor: float in range (0.0, 1.0], optional
            The image scale size, how the data should be scaled to.
        gray_scale: Boolean, optional
            Whether we scale the image to gray or not.
        serialized_sequence_length: int, optional
            The sequence length of each serialized file.
        do_distortion: Boolean, optional
            Whether image distortion should be performed or not.
            Can have a very bad influence on performance.
        crop_size: tuple(int) or None, optional
            The size (height, width) to randomly crop the images.
        """
        image_size = [int(FRAME_HEIGHT * image_scale_factor),
                      int(FRAME_WIDTH * image_scale_factor),
                      1 if gray_scale else FRAME_CHANNELS]
        
        if crop_size is not None:
            assert image_size[0] > crop_size[0] and image_size[1] > crop_size[1], \
                "Image size has to be larger than the crop size."
        
        self._serialized_sequence_length = serialized_sequence_length
        self._do_distortion = do_distortion
        self._crop_size = crop_size
        self._data_img_size = image_size
        
        # download and extract data
        rar_path = light.utils.data.download(UCF11_URL, data_dir)
        dataset_path = light.utils.data.extract(rar_path, data_dir)
        self._data_dir = dataset_path
        
        # generate frame sequences.
        video_filenames = light.utils.path.get_filenames(dataset_path, '*.mpg')
        dataset_size, seq_files = light.utils.data.preprocess_videos(dataset_path, SUBDIR_SHARED,
                                                                  video_filenames,
                                                                  [FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS],
                                                                  serialized_sequence_length,
                                                                  gray_scale, image_scale_factor)
        
        self._file_name_list = seq_files
        self._indices = range(dataset_size)
        self._row = 0
        
        if crop_size is None:
            input_shape = [input_seq_length, image_size[0], image_size[1], image_size[2]]
            target_shape = [target_seq_length, image_size[0], image_size[1], image_size[2]]
        else:
            input_shape = [input_seq_length, crop_size[0], crop_size[1], image_size[2]]
            target_shape = [target_seq_length, crop_size[0], crop_size[1], image_size[2]]
        
        super(UCF11ValidDataset, self).__init__(data_dir, dataset_size, input_shape, target_shape)

    @light.utils.attr.override
    def get_batch(self, batch_size):
        if self._row + batch_size >= self.size:
            self.reset()
        start = self._row
        end = start + batch_size
        ind_range = self._indices[start:end]
        self._row += batch_size
        
        # get next filenames
        file_names = [self._file_name_list[i] for i in ind_range]
        
        # do equal random crop?
        if self._crop_size is not None:
            offset_x = random.randint(0, self._data_img_size[1] - self._crop_size[1])
            offset_y = random.randint(0, self._data_img_size[0] - self._crop_size[0])
        
        # do equal random flip?
        do_flip = False
        if self.do_distortion:
            if random.random() > 0.5:
                do_flip = True
        
        # load serialized sequences
        seq_input_list = []
        seq_target_list = []
        for f in file_names:
            current = light.utils.image.read_as_binary(f, dtype=np.uint8)
            current = np.reshape(current, [self.serialized_sequence_length] + list(self._data_img_size))
            
            # select random part of the sequence with length of inputs+targets
            inputs_length = self.input_shape[0]
            targets_length = self.target_shape[0]
            total_length = inputs_length + targets_length
            start_t = random.randint(0, self.serialized_sequence_length - total_length)
            current = current[start_t:(start_t + total_length)]
            
            if self._crop_size is not None:
                current = current[:, offset_y:(offset_y+self._crop_size[0]),
                                  offset_x:(offset_x+self._crop_size[1]),:]
            
            if do_flip:
                #current = np.flip(current, axis=-2) # only available in numpy 1.1.12 dev0
                current = current[:,:,::-1,:] # horizontal flip
            
            seq_input_list.append(current[0:inputs_length])
            seq_target_list.append(current[inputs_length:(inputs_length + total_length)])
        
        input_sequence = np.stack(seq_input_list)
        target_sequence = np.stack(seq_target_list)
        
        # convert to float of scale [0.0, 1.0]
        inputs = input_sequence.astype(np.float32) / 255
        targets = target_sequence.astype(np.float32) / 255
        
        return inputs, targets
    
    @light.utils.attr.override
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