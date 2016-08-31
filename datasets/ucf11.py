import sys
import os

import numpy as np
import tensorflow as tf
import tensortools as tt
import base


UCF11_URL = 'http://crcv.ucf.edu/data/UCF11_updated_mpg.rar'
SERIALIZED_SEQ_LENGTH = 30
MIN_FRACTION_EXAMPLES_IN_QUEUE = 0.1

FRAME_HEIGHT = 240
FRAME_WIDTH = 320  # TODO: remove these constants???
FRAME_CHANNELS = 3

DO_DISTORTION = False # Activating this is super slow! (factor 8-10)


class UCF11TrainDataset(base.AbstractImageSequenceDataset):
    """UCF-11 sports dataset that creates a bunch of binary frame sequences
       and uses a file queue for multi-threaded input reading.
    """
    def __init__(self, input_seq_length=10, target_seq_length=10,
                 image_size=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)):
        """Creates a dataset instance.
        Parameters
        ----------
        ... TODO: describe parameters of this classes.
        """
        try:
            rar_path = tt.utils.data.download(UCF11_URL, 'tmp')
            dataset_path = tt.utils.data.extract(rar_path, 'tmp')
            self._data_dir = dataset_path
           
        except:
            print 'Please set the correct path to UCF11 dataset. Might be caused by a download error.'
            sys.exit()
        
        dataset_size = self._serialize_frame_sequences(dataset_path, image_size)
        data = None
        
        super(UCF11TrainDataset, self).__init__(data, dataset_size, image_size,
                                                     input_seq_length, target_seq_length)

    @staticmethod
    def _serialize_frame_sequences(dataset_path, image_size):
        sequence_files = tt.utils.path.get_filenames(dataset_path, '*.seq')
        
        if len(sequence_files) > 0:
            dataset_size = len(sequence_files)
            print("Found {} serialized frame sequences. Skipping serialization.". format(dataset_size))
            return dataset_size

        video_filenames = tt.utils.path.get_filenames(dataset_path, '*.mpg')
        frame_scale_factor = image_size[0] / float(FRAME_HEIGHT)
        seq_counter = 0
        for i, video_filename in enumerate(video_filenames):
            with tt.utils.video.VideoReader(video_filename) as vr:
                frames = []
                for f in xrange(SERIALIZED_SEQ_LENGTH):
                    frame = vr.next_frame(frame_scale_factor)
                    if frame is not None:
                        # ensure frame is not too large
                        h, w, c = np.shape(frame)
                        if h > image_size[0] or w > image_size[1]:
                            frame = frame[:image_size[0], :image_size[1], :]
                        if not h < image_size[0] and not w < image_size[1]:
                            frame = np.reshape(frame, [image_size[0], image_size[1], -1]) # TODO: no -1 here necessary
                            if image_size[2] == 1:
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
        print('Successfully extracted {} frame sequences.'.format(seq_counter))
        return seq_counter
    
    def _read_record(self, filename_queue):
        
        class FrameSeqRecord(object):
            pass
        
        record = FrameSeqRecord()
        record.height = self.image_size[0]
        record.width = self.image_size[1]
        record.depth = self.image_size[2]

        frame_bytes = record.height * record.width * record.depth
        total_seq_length = self.input_seq_length + self.target_seq_length
        record_bytes = frame_bytes * (self.input_seq_length + self.target_seq_length)
        total_file_bytes = frame_bytes * SERIALIZED_SEQ_LENGTH

        reader = tf.FixedLengthRecordReader(total_file_bytes)

        record.key, value = reader.read(filename_queue)
        decoded_record_bytes = tf.decode_raw(value, tf.uint8)

        record.data = decoded_record_bytes[0:self.input_seq_length]

        decoded_record_bytes = tf.reshape(decoded_record_bytes,
                                          [SERIALIZED_SEQ_LENGTH, record.height, record.width, record.depth])

        # calculcate tensors [start, 0, 0, 0] and [start + INPUT_SEQ_LENGTH, 0, 0, 0]
        rnd_start_index = tf.to_int32(tf.random_uniform([1], 0, SERIALIZED_SEQ_LENGTH - (total_seq_length), tf.int32))
        seq_start_offset = tf.SparseTensor(indices=[[0]], values=rnd_start_index, shape=[4])
        sequence_start = tf.sparse_tensor_to_dense(seq_start_offset)

        # take first frames as input
        record.data = tf.cast(tf.slice(decoded_record_bytes, sequence_start,
                                       [total_seq_length, record.height, record.width, record.depth]),
                              tf.float32)
        return record

    def get_batch(self, batch_size):
        """Construct input using the Reader ops.
        Args:
            data_dir: Path to the data directory.
            batch_size: Number of image sequences per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, FRAME_HEIGHT, FRAME_WIDTH, 3 * INPUT_SEQ_LENGTH] size.
        """
        seq_filenames = tt.utils.path.get_filenames(self._data_dir, '*.seq')
        with tf.name_scope('preprocessing'):
            filename_queue = tf.train.string_input_producer(seq_filenames)
            seq_record = self._read_record(filename_queue)  

            reshaped_seq = tf.cast(seq_record.data, tf.float32)
            reshaped_seq = (reshaped_seq - 127.5) / 127.5
    
            # distort images
            if DO_DISTORTION:
                with tf.name_scope('distort_inputs'):
                    images_to_distort = []
                    for i in xrange(self.input_seq_length + self.target_seq_length):
                        images_to_distort.append(reshaped_seq[i,:,:,:])

                    distorted_images = tt.image.equal_random_distortion(images_to_distort)
                    stacked_distorted_inputs = tf.pack(distorted_images[0:self.input_seq_length], axis=0)
                    stacked_distorted_targets = tf.pack(distorted_images[self.input_seq_length:], axis=0)
            else:
                stacked_distorted_inputs = reshaped_seq[0:self.input_seq_length,:,:,:]
                stacked_distorted_targets = reshaped_seq[self.input_seq_length:,:,:,:]

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = int(self.size * MIN_FRACTION_EXAMPLES_IN_QUEUE)

        # Generate a batch of sequences and labels by building up a queue of examples.
        batch = tt.inputs.generate_batch(stacked_distorted_inputs, stacked_distorted_targets,
                                        batch_size, min_queue_examples, 
                                        shuffle=True, num_threads=8)
        return batch
