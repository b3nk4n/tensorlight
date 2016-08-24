import os
import time
import collections
from datetime import datetime
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensortools as tt

TRAIN_DIR = 'train'
GPU_ALLOW_GROWTH = True
GPU_MEMORY_FRACTION = 1.0
MAX_STEPS = 2000

# retrieve this from input-data, or model?
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
FRAME_CHANNELS = 1
OUTPUT_SEQ_LENGTH = 10
INPUT_SEQ_LENGTH = 10

EPOCHS = 3 # ?

NUM_GPUS = 2
FIXED_NUM_STEPS_PER_DECAY = 10000
NUM_EPOCHS_PER_DECAY = 75.0 # used if FIXED_NUM_STEPS_PER_DECAY is None

INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_FACTOR = 0.5


class AbstractRuntime(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self._session = None
        self._datasets = collections.namedtuple("datasets", ("train", "valid", "test"))
        self._model_init_func = None
        
        
    def register_datasets(self, train_ds, valid_ds=None, test_ds=None):
        self._datasets.train = train_ds
        self._datasets.valid = valid_ds
        self._datasets.test = test_ds
        
        
    def register_model(self, model_init_func):
        self._model_init_func = model_init_func
        
        
    def start_training(self):
        global_step = tf.Variable(0, trainable=False)
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        # FIXME: placeholders not required, when datasets uses input_queue
        x = tf.placeholder(tf.float32, [None, INPUT_SEQ_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS], "X")
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_SEQ_LENGTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS], "Y_")
        
        train_op, total_loss, loss, summaries = self._train(x, y_, global_step, is_training)
        
        # Create a saver and merge all summaries
        saver = tf.train.Saver(tf.all_variables())
        
        #summary_op = tf.merge_all_summaries()
        summary_op = tf.merge_summary(summaries)
        
        with self.session as sess:
            
            # Initialize all variables
            sess.run(tf.initialize_all_variables())
            
            # Start input enqueue threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)

            self.datasets.train.reset()

            try:
                step = 0
                while not coord.should_stop():
                    step += 1

                    if (step > MAX_STEPS):
                        break

                    start_time = time.time()

                    #batch_x, batch_y = self.datasets.train.get_batch() # TODO: MovingMNIST has to return tuple!
                    batch = self.datasets.train.get_batch()
                    batch_x = batch[:,0:INPUT_SEQ_LENGTH,:,:,:]
                    batch_y = batch[:,INPUT_SEQ_LENGTH:INPUT_SEQ_LENGTH+OUTPUT_SEQ_LENGTH,:,:,:]

                    _, total_loss_value, loss_value = sess.run([train_op, total_loss, loss],
                                             feed_dict={x: batch_x, y_: batch_y, is_training: True})
                    duration = time.time() - start_time

                    assert not np.isnan(total_loss_value), 'Model diverged with cost = NaN'

                    if step % 10 == 0:
                        # info
                        num_examples_per_step = self.datasets.train.batch_size * NUM_GPUS
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = ('%s: step %d, total-loss = %.2f, loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (datetime.now().time(), step,
                                            total_loss_value, loss_value,
                                            examples_per_sec, sec_per_batch))

                    if step % 100:
                        # summary
                        summary_str = sess.run(summary_op, feed_dict={x: batch_x, y_: batch_y, is_training:True})
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush() 

                    if step % 1000 == 0 or step == 100:
                        # validate
                        self.validate()

                    if step % 1000 == 0:
                        # save checkpoint
                        checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                    
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop
                coord.request_stop()

            # Wait for threads to finish
            coord.join(threads)
                
                
    @abstractmethod
    def _train(self, x, y_, global_step, is_training):
        pass
    
    
    def validate(self):
        # TODO: session, when executed alone?
        self.datasets.valid.reset()
        num_batches = self.datasets.valid.dataset_size // self.datasets.valid.batch_size

        valid_loss_sum = 0
        print("Validation...")
        for b in xrange(num_batches):
            batch = self.datasets.valid.get_batch()      
            batch_x = batch[:,0:INPUT_SEQ_LENGTH,:,:,:]
            batch_y = batch[:,INPUT_SEQ_LENGTH:INPUT_SEQ_LENGTH+OUTPUT_SEQ_LENGTH,:,:,:]

            valid_loss = sess.run(loss, feed_dict={x: batch_x, y_: batch_y, is_training: False})
            print("{}/{} Valid Loss= {:.6f}".format(b, num_batches, valid_loss))
            valid_loss_sum += valid_loss
            
        avg_valid_loss = valid_loss_sum / num_batches
        print("@{}: Minibatch Avg. Valid Loss= {:.6f}".format(step, avg_valid_loss))
        valid_loss_summary = tf.scalar_summary('avg_valid_loss', avg_valid_loss)
        summary_str = sess.run(valid_loss_summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush() 
        
        
    def test(self):
        pass
        
        

    @property
    def session(self):
        if self._session is None:
            gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION,
            allow_growth=GPU_ALLOW_GROWTH)
            self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        return self._session
    
    @property
    def datasets(self):
        return self._datasets

    
    
    
    
class DefaultRuntime(AbstractRuntime):
    
    def __init__(self):
        super(AbstractRuntime, self).__init__()
    
    
    
    
    
class MultiGpuRuntime(AbstractRuntime):
    
    def __init__(self):
        super(MultiGpuRuntime, self).__init__()
        

    def _tower_loss(self, model, scope):
        """Calculate the total loss on a single tower.
        Args:
            scope: unique prefix string identifying the tower, e.g. 'tower_0'
        Returns:
            Tensor of shape [] containing the total loss for a batch of data
        """
        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        total_loss = model.total_loss
        loss = model.loss

        # Compute the moving average of all individual losses and the total loss.
        # Generate moving averages of all losses and associated summaries 
        loss_averages_op = tt.board.loss_summary([total_loss, loss] +
                                                 tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) + # FIXME: redundant!??! Is the list empty???
                                                 tf.get_collection('intermediate_losses', scope),
                                                 decay=0.9)

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
            loss = tf.identity(loss)

        return total_loss, loss
        
    #@override
    def _train(self, x, y_, global_step, is_training):
        """Train sequence model.
        predictions_for_input:
            Either the same as predictions or None
        """
        # Variables that affect learning rate
        batch_size = self.datasets.train.batch_size
        num_batches_per_epoch = self.datasets.train.dataset_size / batch_size
        decay_steps = num_batches_per_epoch * NUM_EPOCHS_PER_DECAY

        if FIXED_NUM_STEPS_PER_DECAY is not None:
            decay_steps = FIXED_NUM_STEPS_PER_DECAY

        # Decay the learning rate exponentially based on the number of steps
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.scalar_summary('learning_rate', lr)

        # Compute gradients
        opt = tf.train.AdamOptimizer(lr) # TODO: make optimizer parameterizable? or spcify in model?
        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_total_losses = []
        tower_losses = []
        batch_size_per_gpu = (batch_size // NUM_GPUS)
        for i in xrange(NUM_GPUS):
            with tf.device('/gpu:%d' % i, ):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    this_inputs = x[i*batch_size_per_gpu:(i+1)*batch_size_per_gpu, :, :, :, :]
                    this_targets = y_[i*batch_size_per_gpu:(i+1)*batch_size_per_gpu, :, :, :, :]

                    # Build inference Graph.This function constructs 
                    # the entire model but shares the variables across all towers.
                    model = self._model_init_func(this_inputs, this_targets,
                                                  scope=scope, is_training=is_training)

                    # Calculate the loss for one tower of the model.
                    this_total_loss, this_loss = self._tower_loss(model, scope)
                    tower_total_losses.append(this_total_loss)
                    tower_losses.append(this_loss)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this tower.
                    grads = opt.compute_gradients(this_total_loss)

                    # Keep track of the gradients ackeep_probross all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient.
        # This is also the synchronization point across all towers.
        grads = tt.training.average_gradients(tower_grads)

        total_loss = tf.reduce_mean(tower_total_losses)                  
        summaries.append(tf.scalar_summary('mean_total_loss', total_loss))

        loss = tf.reduce_mean(tower_losses)                  
        summaries.append(tf.scalar_summary('mean_loss', loss))

        # Add a summary to track the learning rate.
        summaries.append(tf.scalar_summary('learning_rate', lr))

        # Add histograms for gradients
        summaries.extend(tt.board.gradients_histogram_summary(grads))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        summaries.extend(tt.board.variables_histogram_summary())

        # Track the moving averages of all trainable variables
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op, name="train_op")

        return train_op, total_loss, loss, summaries
        
        
    def start_training(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            return super(MultiGpuRuntime, self).start_training()
            