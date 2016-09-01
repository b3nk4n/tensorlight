import os
import time
import collections
from datetime import datetime
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensortools as tt

TRAIN_DIR = 'train' # (automatially train_<ModelClassName> ?)
MAX_CHECKPOINTS_TO_KEEP = 5 # (default = 5, None/0 == all)
GPU_ALLOW_GROWTH = True # fixed? :)
GPU_MEMORY_FRACTION = 1.0 # fixed? :)

FIXED_NUM_STEPS_PER_DECAY = 10000
NUM_EPOCHS_PER_DECAY = 75.0 # used if FIXED_NUM_STEPS_PER_DECAY is None <-- UNUSED; Due to batch_size refactoring

INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.5

LATEST_CHECKPOINT = 'LATEST'

class AbstractRuntime(object):
    __metaclass__ = ABCMeta
    
    def __init__(self):
        self._graph = tf.Graph()
        with self.graph.as_default():
            self._session = None
            self._datasets = collections.namedtuple("datasets", ("train", "valid", "test"))
            self._model = None
            self._inferences = []

            self._coord = None
            self._threads = None

            self._feed_func = None
            self._x = None
            self._y = None

            self._batch_size_ph = None

            self._is_training = tf.placeholder(tf.bool, name='is_training')
            self._global_step = tf.Variable(0, trainable=False)
            self._batch_size_ph = tf.placeholder(tf.int32, name='batch_size')
            self._input_from_queue = tf.placeholder(tf.bool, name='input_from_queue')

            self._saver = None
            self._summary_writer = None

            self._train_op = None
            self._total_loss = None
            self._loss = None
            self._summary_op = None
           
    def register_datasets(self, train_ds, valid_ds=None, test_ds=None):
        self._datasets.train = train_ds
        self._datasets.valid = valid_ds
        self._datasets.test = test_ds
        
    def register_model(self, model):
        self._model = model
        
    def build(self, is_autoencoder=False, checkpoint_file=None):
        """ checkpoint_file: file-name in TRAIN_DIR, or 'latest' """
        with self.graph.as_default():
            self._x = tf.placeholder(tf.float32, [None] + self._datasets.train.input_shape, "X")
            if is_autoencoder:
                self._y = tf.placeholder(tf.float32, [None] + self._datasets.train.input_shape, "Y")
            else:
                self._y = tf.placeholder(tf.float32, [None] + self._datasets.train.target_shape, "Y")

            if self._datasets.train.uses_queue:
                inputs, targets = self._datasets.train.get_batch(self._batch_size_ph)
                if is_autoencoder:
                    targets = inputs
            else:
                # we have to assign these to have their tensor shape equal,
                # even if this is never evaluated by tf.cond().
                inputs = self._x
                targets = self._y

            x = tf.cond(self._input_from_queue, lambda: inputs, lambda: self._x)
            y = tf.cond(self._input_from_queue, lambda: targets, lambda: self._y)


            if is_autoencoder:
                self._feed_func = lambda inputs, targets, bs, is_train: {self._x: inputs,
                                                                         self._y: inputs,
                                                                         self._batch_size_ph: bs,
                                                                         self._is_training: is_train}
            else:
                self._feed_func = lambda inputs, targets, bs, is_train : {self._x: inputs,
                                                                          self._y: targets,
                                                                          self._batch_size_ph: bs,
                                                                          self._is_training: is_train}

            train_op, total_loss, loss, summaries = self._build_internal(x, y)

            self._train_op = train_op
            self._total_loss = total_loss
            self._loss = loss
            if summaries is None:
                self._summary_op = tf.merge_all_summaries()
            else:
                self._summary_op = tf.merge_summary(summaries)
                
            # Create a saver to store checkpoints of the model
            self._saver = tf.train.Saver(max_to_keep=MAX_CHECKPOINTS_TO_KEEP)

            if checkpoint_file is None:
                # start session and init all variables
                self.session.run(tf.initialize_all_variables())
            else:
                if checkpoint_file == LATEST_CHECKPOINT:
                    checkpoint_path = tf.train.latest_checkpoint(TRAIN_DIR)
                    assert checkpoint_path is not None, "No latest checkpoint file found."
                else:
                    checkpoint_path = os.path.join(TRAIN_DIR, checkpoint_file)
                self._saver.restore(self.session, checkpoint_path)
                print("Restoring variables from: {}".format(checkpoint_path))

            # creates coordinatior and queue threads
            self._coord = tf.train.Coordinator()
            if self.datasets.train.uses_queue:
                self._threads = tf.train.start_queue_runners(sess=self.session, coord=self._coord)
                
            params_count = tt.core.trainable_parameters_count()
            print("Total model-params: {}".format(params_count))
        
    def train(self, batch_size, steps=-1, epochs=-1, display_step=10):
        assert not(steps <= 0 and epochs <= 0), "Either set 'steps' or 'epochs' parameter"
        assert not(steps > 0 and epochs > 0), "Not allowed to set both, 'steps' and 'epochs' parameter"
        
        with self.graph.as_default():
            batches_per_epoch = self.datasets.train.size // batch_size

            if epochs > 0:
                steps = batches_per_epoch * epochs

            # TODO: this is only required for inpute Queue!!!
            # Start input enqueue threads
            #coord = tf.train.Coordinator()

            #if self.datasets.train.uses_queue:
            #    threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
            #else:
            #    threads = None

            self.datasets.train.reset()

            try:
                this_step = 0
                total_loss_sum = 0
                loss_sum = 0

                x_dummy = np.zeros([batch_size] + self.datasets.train.input_shape)
                y_dummy = np.zeros([batch_size] + self.datasets.train.target_shape)

                while not self._coord.should_stop():
                    this_step += 1
                    if (this_step > steps):
                        break

                    if this_step % batches_per_epoch == 1:
                        epoch = (this_step - 1) // batches_per_epoch + 1
                        print("Starting epoch {}...".format(epoch))

                    start_time = time.time()

                    if self.datasets.train.uses_queue:
                        batch_x = x_dummy
                        batch_y = y_dummy
                    else:
                        batch_x, batch_y = self.datasets.train.get_batch(batch_size)
                    feed = self._feed_func(batch_x, batch_y, batch_size, True)
                    feed.update({self._input_from_queue: True if self.datasets.train.uses_queue else False})

                    if this_step == 1 and self.datasets.train.uses_queue:
                        print("Filling queue with {} examples...".format(-1)) # TODO: retrieve real number...

                    # step counter is increment when train_op is executed
                    pred, _, gstep, total_loss, loss = self.session.run([self._inferences[0], self._train_op,
                                                                  self._global_step,
                                                                  self._total_loss,
                                                                  self._loss],
                                                                 feed_dict=feed)
                    duration = time.time() - start_time

                    assert not np.isnan(loss), 'Warning: Model diverged with loss = NaN'

                    total_loss_sum += total_loss
                    loss_sum += loss
                    if gstep % display_step == 0:
                        # info
                        num_examples_per_step = batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        avg_total_loss = total_loss_sum / display_step
                        avg_loss = loss_sum / display_step
                        total_loss_sum = 0
                        loss_sum = 0
                        print("@{:6d}: loss: {:9.3f}, total-loss: {:9.3f} ({:7.1f} examples/sec, {:5.2f} sec/batch)" \
                              .format(gstep, avg_loss, avg_total_loss, examples_per_sec, sec_per_batch))

                    if gstep % 100 or this_step == steps:
                        # summary
                        summary_str = self.session.run(self._summary_op, feed_dict=feed)
                        self.summary_writer.add_summary(summary_str, gstep)
                        self.summary_writer.flush() 

                    if gstep == 100 or this_step == steps or epochs == -1 and gstep % 1000 == 0 or \
                       epochs > 0 and this_step % batches_per_epoch == 0:
                        # validate
                        print
                        self._test_internal(batch_size, self.datasets.valid, "validation", True)
                        print

                    if gstep % 1000 == 0 or this_step == steps:
                        # save regular checkpoint
                        checkpoint_path = os.path.join(TRAIN_DIR, "model.ckpt")
                        self._saver.save(self.session, checkpoint_path, global_step=self._global_step)

                    if epochs > 0:
                        if this_step % batches_per_epoch == 0:
                            # save epoch checkpoint
                            checkpoint_path = os.path.join(TRAIN_DIR, "ep-{}_model.ckpt".format(epoch))
                            self._saver.save(self.session, checkpoint_path, global_step=self._global_step)

            except tf.errors.OutOfRangeError:
                print("Done training -- epoch limit reached")
            #finally:
                # When done, ask the threads to stop
                #self._coord.request_stop()
                #print("Coordinator stopped.")

            # Wait for threads to finish
            #if self._threads is not None:
                #self._coord.join(self._threads)

    @abstractmethod
    def _build_internal(self, x, y):
        pass
    
    def predict(self, inputs): 
        with self.graph.as_default():
            feed = self._feed_func(inputs, None, inputs.shape[0], False)
            feed.update({self._input_from_queue: False})
            return self.session.run(self._inferences[0], feed_dict=feed)
        
    def validate(self, batch_size):
        with self.graph.as_default():
            self._test_internal(batch_size, self.datasets.valid, "validation")
             
    def test(self, batch_size):
        with self.graph.as_default():
            self._test_internal(batch_size, self.datasets.test, "test")
           
    def _test_internal(self, batch_size, dataset, title, summary=False):
        if dataset is None:
            print("No {} dataset registered. Skipping.".format(title))
            return
        
        batches_per_epoch = dataset.size // batch_size
        num_batches = batches_per_epoch
        gstep = self.session.run(self._global_step)
        print("@{:6d}: Starting {} (batch-size: {}, dataset-size: {}):" \
              .format(gstep, title, batch_size, dataset.size))
        
        dataset.reset()
        loss_sum = 0
        x_dummy = np.zeros([batch_size] + dataset.input_shape)
        y_dummy = np.zeros([batch_size] + dataset.target_shape)
        progress = tt.utils.ui.ProgressBar(num_batches * batch_size)
        for b in xrange(num_batches):
            if self.datasets.train.uses_queue:
                batch_x = x_dummy
                batch_y = y_dummy
            else:
                batch_x, batch_y = dataset.get_batch(batch_size)
            feed = self._feed_func(batch_x, batch_y, batch_size, False)
            feed.update({self._input_from_queue: True if dataset.uses_queue else False})

            this_loss = self.session.run(self._loss, feed_dict=feed)
            loss_sum += this_loss
            progress.update((b+1) * batch_size, [('loss', this_loss)])
            
        if summary:
            avg_loss = loss_sum / num_batches
            loss_summary = tf.scalar_summary("{}_loss".format(title), avg_loss)
            summary_str = self.session.run(loss_summary)
            self.summary_writer.add_summary(summary_str, gstep)
            self.summary_writer.flush()
    
    def _create_session(self):
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION,
            allow_growth=GPU_ALLOW_GROWTH)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
    def close(self):
        self._coord.request_stop()
        print("Coordinator stopped.")

        # Wait for threads to finish
        if self._threads is not None:
            self._coord.join(self._threads)
        
        self.session.close()
        
    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        if self._session is None:
            self._session = self._create_session()
        return self._session
    
    @property
    def datasets(self):
        return self._datasets

    @property
    def summary_writer(self):
        if self._summary_writer is None:
            self._summary_writer = tf.train.SummaryWriter(TRAIN_DIR, self.session.graph)
        return self._summary_writer
    
    
    
class DefaultRuntime(AbstractRuntime):
    
    def __init__(self):
        print("Launing default runtime...")
        super(DefaultRuntime, self).__init__()
        
    @tt.utils.attr.override
    def _build_internal(self, x, y):
        # Variables that affect learning rate
        #batch_size = self.datasets.train.batch_size
        #num_batches_per_epoch = self.datasets.train.size / batch_size
        #decay_steps = num_batches_per_epoch * NUM_EPOCHS_PER_DECAY

        if FIXED_NUM_STEPS_PER_DECAY is not None:
            decay_steps = FIXED_NUM_STEPS_PER_DECAY

        # Decay the learning rate exponentially based on the number of steps
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        self._global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.scalar_summary('learning_rate', lr)
        
        # Build inference Graph.This function constructs 
        # the entire model but shares the variables across all towers.
        inference = self._model.inference(x, y,
                                          is_training=self._is_training)
        self._inferences.append(inference)
        
        loss = self._model.loss(inference, y)
        total_loss = self._model.total_loss(loss, inference, y)

        # Generate moving averages of all losses and associated summaries
        loss_averages_op = tt.board.loss_summary([total_loss, loss] +
                                                 tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) + # FIXME: redundant!??! Is the list empty???
                                                 tf.get_collection('intermediate_losses'),
                                                 decay=0.9)

        # Compute gradients
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients
        apply_gradient_op = opt.apply_gradients(grads, global_step=self._global_step)

        tt.board.variables_histogram_summary()

        # Add histograms for gradients
        tt.board.gradients_histogram_summary(grads)

        # Track the moving averages of all trainable variables
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, self._global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op, name="train_op")
        return train_op, total_loss, loss, None # FIXME: collect summaries above? Return get_all_sum?
        


class MultiGpuRuntime(AbstractRuntime):
    
    def __init__(self, num_gpus=2):
        print("Launing Multi-GPU runtime...")
        
        device_list = tt.hardware.get_cuda_devices()
        if len(device_list) == 0:
            print("Selecting all GPU devices.")
        else:
            assert len(device_list) >= num_gpus, "Not enough GPU devices available."
            print("Selecting GPU devices: {}".format(device_list[0:num_gpus]))
        self._num_gpus = num_gpus
        
        super(MultiGpuRuntime, self).__init__()
        
    @tt.utils.attr.override
    def _build_internal(self, x, y):
        # Variables that affect learning rate
        #batch_size = self.datasets.train.batch_size
        #num_batches_per_epoch = self.datasets.train.size / batch_size
        #decay_steps = num_batches_per_epoch * NUM_EPOCHS_PER_DECAY

        if FIXED_NUM_STEPS_PER_DECAY is not None:
            decay_steps = FIXED_NUM_STEPS_PER_DECAY

        # Decay the learning rate exponentially based on the number of steps
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        self._global_step,
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
        splitted_x = tf.split(0, self.num_gpus, x)
        splitted_y = tf.split(0, self.num_gpus, y)
        for i in xrange(self.num_gpus):
            with tf.device('/gpu:%d' % i, ):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    this_inputs = splitted_x[i]
                    this_targets = splitted_y[i]

                    # Build inference Graph.This function constructs 
                    # the entire model but shares the variables across all towers.
                    inference = self._model.inference(this_inputs, this_targets,
                                                      is_training=self._is_training,
                                                      device_scope=scope,
                                                      memory_device='/cpu:0')
                    self._inferences.append(inference)
                    
                    loss = self._model.loss(inference, this_targets)
                    total_loss = self._model.total_loss(loss, inference, this_targets)

                    # Calculate the moving averages of the loss for one tower of the model
                    loss_averages_op = tt.board.loss_summary([total_loss, loss] +
                                                             tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) + # FIXME: redundant!??! Is the list empty???
                                                             tf.get_collection('intermediate_losses', scope),
                                                             decay=0.9)

                    with tf.control_dependencies([loss_averages_op]):
                        this_total_loss = tf.identity(total_loss)
                        this_loss = tf.identity(loss)

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
        summaries.append(tf.scalar_summary('mean_total_loss', total_loss)) # TODO: really needed?

        loss = tf.reduce_mean(tower_losses)                  
        summaries.append(tf.scalar_summary('mean_loss', loss)) # TODO: really needed?

        # Add a summary to track the learning rate.
        summaries.append(tf.scalar_summary('learning_rate', lr))

        # Add histograms for gradients
        summaries.extend(tt.board.gradients_histogram_summary(grads))

        # Apply the gradients to adjust the shared variables and increment the step counter
        apply_gradient_op = opt.apply_gradients(grads, global_step=self._global_step)

        summaries.extend(tt.board.variables_histogram_summary())

        # Track the moving averages of all trainable variables
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, self._global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op, name="train_op")
        return train_op, total_loss, loss, summaries
        
    @tt.utils.attr.override
    def train(self, batch_size, steps=-1, epochs=-1, display_step=10):
        assert batch_size % float(self.num_gpus) == 0, "Batch-size has to be multiples of 'num_gpus'."
        with tf.device('/cpu:0'):
            return super(MultiGpuRuntime, self).train(batch_size, steps, epochs, display_step)              
    
    @tt.utils.attr.override
    def validate(self, batch_size):
        assert batch_size % float(self.num_gpus) == 0, "Batch-size has to be multiples of 'num_gpus'."
        return super(MultiGpuRuntime, self).validate(batch_size)
    
    @tt.utils.attr.override
    def test(self, batch_size):
        assert batch_size % float(self.num_gpus) == 0, "Batch-size has to be multiples of 'num_gpus'."
        return super(MultiGpuRuntime, self).test(batch_size)
        
    @property
    def num_gpus(self):
        return self._num_gpus

    
    
def trainable_parameters_count():
    """Gets the number of trainable parameters in this graph.
    Returns
    ----------
    The number of trainable variables (= model parameters).
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters