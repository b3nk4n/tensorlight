import sys
import os

import numpy as np
import tensorflow as tf
import tensortools as tt


UCF11_URL = 'http://crcv.ucf.edu/data/UCF11_updated_mpg.rar'
SERIALIZED_SEQ_LENGTH = 30
MIN_FRACTION_EXAMPLES_IN_QUEUE = 0.15

FRAME_HEIGHT = 240
FRAME_WIDTH = 320
FRAME_CHANNELS = 3



class UCF11TrainDataset(object):
    """UCF-11 sports dataset that creates a bunch of binary frame sequences
       and uses a file queue for multi-threaded input reading.
    """
    def __init__(self, batch_size, num_frames, image_size=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)):
        """Creates a dataset instance.
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        """
        self._seq_length = num_frames
        self._batch_size = batch_size
        self._image_size = image_size
        self._dataset_size = 0 # will be set in frame serialization
        self._frame_size = self._image_size[0] * self._image_size[1] * self._image_size[2]

        try:
            rar_path = tt.utils.data.download(UCF11_URL, 'tmp')
            dataset_path = tt.utils.data.extract(rar_path, 'tmp')
            self._data_dir = dataset_path
           
        except:
            print 'Please set the correct path to UCF11 dataset. Might be caused by a download error.'
            sys.exit()
        
        self._serialize_frame_sequences(dataset_path)
        
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
    
    def _serialize_frame_sequences(self, dataset_path):
        sequence_files = tt.utils.path.get_filenames(dataset_path, '*.seq')
        
        if len(sequence_files) > 0:
            self._dataset_size = len(sequence_files)
            print("Found {} serialized frame sequences. Skipping serialization.". format(self.dataset_size))
            return

        video_filenames = tt.utils.path.get_filenames(dataset_path, '*.mpg')
        print(len(video_filenames))
        frame_scale_factor = int(FRAME_HEIGHT / self._image_size[0])
        seq_counter = 0
        for i, video_filename in enumerate(video_filenames):
            with tt.utils.video.VideoReader(video_filename) as vr:
                frames = []
                for f in xrange(SERIALIZED_SEQ_LENGTH):
                    frame = vr.next_frame(frame_scale_factor)
                    if frame is not None:
                        # ensure frame is not too large
                        h, w, c = np.shape(frame)
                        if h > self._image_size[0] or w > self._image_size[1]:
                            frame = frame[:self._image_size[0], :self._image_size[1], :]
                        if not h < self._image_size[0] and not w < self._image_size[1]:
                            frame = np.reshape(frame, [self._image_size[0], self._image_size[1], -1])
                            if self._image_size[2] == 1:
                                frame = tt.utils.image.to_grayscale(frame)
                            frames.append(frame)
                        else:
                            print('Warning: Frame bounds too small. Skipping.')
                            break
                    else:
                        print('Warning: Frame sequence too short. Skipping.')
                        break

                if len(frames) == SERIALIZED_SEQ_LENGTH:
                    # TODO: seqences from one folder to a single file?
                    seq_filepath = os.path.splitext(video_filename)[0] + '.seq'
                    tt.utils.image.write_as_binary(seq_filepath, np.asarray(frames))
                    seq_counter += 1
        self._dataset_size = seq_counter
        print('Successfully extracted {} frame sequences.'.format(self.dataset_size))
    
    def _read_record(self, filename_queue):
        
        class FrameSeqRecord(object):
            pass
        
        record = FrameSeqRecord()
        record.height = self._image_size[0]
        record.width = self._image_size[1]
        record.depth = self._image_size[2]

        frame_bytes = record.height * record.width * record.depth
        record_bytes = frame_bytes * (self._seq_length + 1)
        total_file_bytes = frame_bytes * SERIALIZED_SEQ_LENGTH

        reader = tf.FixedLengthRecordReader(total_file_bytes)

        record.key, value = reader.read(filename_queue)
        decoded_record_bytes = tf.decode_raw(value, tf.uint8)

        record.data = decoded_record_bytes[0:(self._seq_length )]
        record.prediction = decoded_record_bytes[self._seq_length]

        decoded_record_bytes = tf.reshape(decoded_record_bytes,
                                          [SERIALIZED_SEQ_LENGTH, record.height, record.width, record.depth])

        # calculcate tensors [start, 0, 0, 0] and [start + INPUT_SEQ_LENGTH, 0, 0, 0]
        rnd_start_index = tf.to_int32(tf.random_uniform([1], 0, SERIALIZED_SEQ_LENGTH - (self._seq_length  + 1), tf.int32))
        seq_start_offset = tf.SparseTensor(indices=[[0]], values=rnd_start_index, shape=[4])
        sequence_start = tf.sparse_tensor_to_dense(seq_start_offset)
        pred_start_offset = tf.SparseTensor(indices=[[0]], values=rnd_start_index + self._seq_length , shape=[4])
        prediction_start = tf.sparse_tensor_to_dense(pred_start_offset)

        # take first n-1 frames as input
        record.data = tf.cast(tf.slice(decoded_record_bytes, sequence_start,
                                       [self._seq_length , record.height, record.width, record.depth]),
                              tf.float32)
        # take last frame as prediction
        record.prediction = tf.cast(tf.slice(decoded_record_bytes, prediction_start,
                                             [1, record.height, record.width, record.depth]),
                                    tf.float32)
        record.prediction = tf.squeeze(record.prediction, squeeze_dims=[0])
        return record
    
    def reset(self):
        pass

    def get_batch(self):
        """Construct input using the Reader ops.
        Args:
            data_dir: Path to the data directory.
            batch_size: Number of image sequences per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, FRAME_HEIGHT, FRAME_WIDTH, 3 * INPUT_SEQ_LENGTH] size.
        """
        seq_filenames = tt.utils.path.get_filenames(self._data_dir, '*.seq')
        print(len(seq_filenames))
        with tf.name_scope('preprocessing'):
            filename_queue = tf.train.string_input_producer(seq_filenames)
            seq_record = self._read_record(filename_queue)  

            reshaped_seq = tf.cast(seq_record.data, tf.float32)

            seq_record.prediction = (seq_record.prediction - 127.5) / 127.5
            reshaped_seq = (reshaped_seq - 127.5) / 127.5
    
            # distort images
            with tf.name_scope('distort_inputs'):
                images_to_distort = []
                images_to_distort.append(seq_record.prediction)
                for i in xrange(self._seq_length):
                    images_to_distort.append(reshaped_seq[i,:,:,:])

                distorted_images = tt.image.equal_random_distortion(images_to_distort)
                distorted_prediction = distorted_images[0]
                stacked_distorted_input = tf.concat(2, distorted_images[1:self._seq_length+1])

        # Ensure that the random shuffling has good mixing properties.
        num_examples_per_epoch = self._dataset_size
        min_queue_examples = int(num_examples_per_epoch *
                                 MIN_FRACTION_EXAMPLES_IN_QUEUE)
        print("Filling queue with {} examples...".format(min_queue_examples))
    
        # Generate a batch of sequences and labels by building up a queue of examples.
        return tt.inputs.generate_batch(stacked_distorted_input, distorted_prediction,
                                        self._batch_size, min_queue_examples, 
                                        shuffle=True, num_threads=8)
