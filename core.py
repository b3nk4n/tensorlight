import os
import sys
import time
import collections
import copy
from datetime import datetime
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops
import tensortools as tt


LATEST_CHECKPOINT = 'LATEST'
INTERMEDIATE_LOSSES = 'intermediate_losses'


class AbstractRuntime(object):
    """Abstract runtime."""
    __metaclass__ = ABCMeta
    
    def __init__(self, train_dir='train', max_checkpoints_to_keep=5,
                 gpu_allow_growth=True, gpu_memory_fraction=1.0):
        """Creates a base runtime.
        Parameters
        ----------
        train_dir: str, optional
            The training directory for checkpoints and summary files.
        max_checkpoints_to_keep: int, optional
            The number of last checkpoints to keep. Defaults to 5.
            Use 0 or None to keep all checkpoints.
        gpu_allow_growth: Boolean, optional
            Whether the GPUS is allowed to allocate memory dynamically.
            Has the advantage to only use that much memory as it really needs,
            but the downside of memory fragmentation and probably lower performance.
        gpu_memory_fraction: float in range (0, 1], optional
            The fraction of the (currently available) memory it is allows to reserve.
        """
        assert gpu_memory_fraction > 0 and gpu_memory_fraction <= 1, "GPU memory fraction has to be in range (0,1]."
        
        self._graph = tf.Graph()
        self._session = None
        self._datasets = collections.namedtuple("datasets", ("train", "valid", "test"))
        self._model = None
        self._inferences = []

        self._coord = None
        self._threads = None

        self._feed_func = None
        self._model_feeds = None

        self._saver = None
        self._summary_writer = None

        self._train_op = None
        self._summaries = None
        
        self._total_loss = None
        self._loss = None
            
        with self.graph.as_default():
            self._global_step = tf.Variable(0, trainable=False)
            
            self._ph = collections.namedtuple("placeholders", ("inputs",
                                                               "targets"
                                                               "is_training",
                                                               "batch_size",
                                                               "input_from_queue"))
            self._ph.is_training = tf.placeholder(tf.bool, name='is_training')
            self._ph.batch_size = tf.placeholder(tf.int32, name='batch_size')
            self._ph.input_from_queue = tf.placeholder(tf.bool, name='input_from_queue')
            
        self._train_dir = train_dir
        self._max_checkpoints_to_keep = max_checkpoints_to_keep
        
        self._gpu = collections.namedtuple("placeholders", ("allow_growth", "memory_fraction"))
        self._gpu.allow_growth = gpu_allow_growth
        self._gpu.memory_fraction = gpu_memory_fraction
           
    def register_datasets(self, train_ds, valid_ds=None, test_ds=None):
        """Registers the datasets.
        Parameters
        ----------
        train_ds: AbstractDataset
            The training dataset.
        valid_ds: AbstractDataset, optional
            The validation dataset.
        test_ds: AbstractDataset, optional
            The test dataset.
        """
        self._datasets.train = train_ds
        self._datasets.valid = valid_ds
        self._datasets.test = test_ds
        
    def register_model(self, model):
        """Registers the model.
        Parameters
        ----------
        model: AbstractModel
            The model to register.
        """
        self._model = model
        
    def build(self, initial_lr, lr_decay_step_interval=sys.maxint, lr_decay_factor=1.0, lr_decay_staircase=True,
              is_autoencoder=False, checkpoint_file=None, verbose=False):
        """ Builds the model. This must be calles before training, validation, testing or prediction.
        Parameters
        ----------
        initial_lr: float
            The initial learning rate, that will use exponential decay.
        lr_decay_step_interval: int, optional
            The step rate when to perform learning rate decay.
            Defaults to sys.maxint, equivalent to no decay.
        lr_decay_factor: float, optional
            The learning rate decay factor.
        lr_decay_staircase: Boolean, optional
            Whether we use smooth decay or staircase decay (default).
        is_autoencoder: Boolean, optional
            Whether we build an autoencoder, where the inputs equals the targets.
        checkpoint_file: str, optional
            The filename of the checkpoint file withing 'train_dir' to restore.
            Use 'LATEST' or 'tt.core.LATEST_CHECKPOINT' to restore the lastest file.
        verbose: Boolean, optional
            Set to True to show additional construction/variable information.
        """
        with self.graph.as_default():
            self._ph.inputs = tf.placeholder(tf.float32, [None] + self._datasets.train.input_shape, "X")
            if is_autoencoder:
                self._ph.targets = tf.placeholder(tf.float32, [None] + self._datasets.train.input_shape, "Y")
            else:
                self._ph.targets = tf.placeholder(tf.float32, [None] + self._datasets.train.target_shape, "Y")

            if isinstance(self.datasets.train, tt.datasets.base.AbstractQueueDataset):
                inputs, targets = self._datasets.train.get_batch(self._ph.batch_size)
                if is_autoencoder:
                    targets = inputs
            else:
                # we have to assign these to have their tensor shape equal,
                # even if this is never evaluated by tf.cond().
                inputs = self._ph.inputs
                targets = self._ph.targets

            with tf.name_scope("feed_or_queue"):
                x = tf.cond(self._ph.input_from_queue, lambda: inputs, lambda: self._ph.inputs)
                y = tf.cond(self._ph.input_from_queue, lambda: targets, lambda: self._ph.targets)

                if is_autoencoder:
                    self._feed_func = lambda inputs, targets, bs, is_train: {self._ph.inputs: inputs,
                                                                             self._ph.targets: inputs,
                                                                             self._ph.batch_size: bs,
                                                                             self._ph.is_training: is_train}
                else:
                    self._feed_func = lambda inputs, targets, bs, is_train : {self._ph.inputs: inputs,
                                                                              self._ph.targets: targets,
                                                                              self._ph.batch_size: bs,
                                                                              self._ph.is_training: is_train}
                self._model_feeds = self._model.fetch_feeds();

            # Decay the learning rate exponentially based on the number of steps
            lr = tf.train.exponential_decay(initial_lr,
                                            self._global_step,
                                            lr_decay_step_interval,
                                            lr_decay_factor,
                                            staircase=lr_decay_staircase)
            
            opt = tf.train.AdamOptimizer(lr)
              
            # build (multi-)device specific computation graph for inference
            grads, summaries, total_loss, loss = self._build_computation_graph(x, y, opt)
            
            # Apply gradients
            apply_gradient_op = opt.apply_gradients(grads, global_step=self._global_step)

            # Add summaries
            summaries.append(tf.scalar_summary('learning_rate', lr))
            summaries.extend(tt.board.gradients_histogram_summary(grads))
            summaries.extend(tt.board.variables_histogram_summary())

            # Track the moving averages of all trainable variables
            variable_averages = tf.train.ExponentialMovingAverage(0.9999, self._global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            train_op = tf.group(apply_gradient_op, variables_averages_op, name="train_op")
            
            # fetch update ops, that is required e.g. for tf.contrib.layers.batch_norm
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                if verbose:
                    print("Found {} update ops.".format(len(update_ops)))
                updates = tf.tuple(update_ops)
                total_loss = control_flow_ops.with_dependencies(updates, total_loss)

            self._train_op = train_op
            self._summaries = summaries
            self._total_loss = total_loss
            self._loss = loss
                
            # Create a saver to store checkpoints of the model
            self._saver = tf.train.Saver(max_to_keep=self.max_checkpoints_to_keep)

            if checkpoint_file is None:
                # start session and init all variables
                self.session.run(tf.initialize_all_variables())
            else:
                if checkpoint_file == LATEST_CHECKPOINT:
                    checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
                    assert checkpoint_path is not None, "No latest checkpoint file found."
                else:
                    checkpoint_path = os.path.join(self.train_dir, checkpoint_file)
                self._saver.restore(self.session, checkpoint_path)
                print("Restoring variables from: {}".format(checkpoint_path))

            # creates coordinatior and queue threads
            self._coord = tf.train.Coordinator()
            if isinstance(self.datasets.train, tt.datasets.base.AbstractQueueDataset):
                self._threads = tf.train.start_queue_runners(sess=self.session, coord=self._coord)
                
            # Model information
            tt.core.show_trainable_parameters(verbose)
            
    @abstractmethod
    def _build_computation_graph(self, x, y, opt):
        """Builds the (device or runtime specific) computation graph.
        Parameters
        ----------
        x: n-D Tensor
            The inputs tensor.
        y: m-D Tensor
            The targets tensor.
        opt: Optimizer
            The TensorFlow optimizer instance.
        Returns
        ----------
        A tuple of (grads, summaries, total_loss, loss)
        """
        pass
        
    def train(self, batch_size, steps=-1, epochs=-1, train_feeds={}, valid_feeds={},
              on_validate=None, display_step=10, summary_steps=100, checkpoint_steps=1000,
              validation_steps=1000, early_validation_at_step=100,
              do_checkpoints=True, do_summary=True):
        """Train the model.
           Note that either 'steps' or 'epochs' has to be defined as a param.
        Parameters
        ----------
        batch_size: int
            The batch size to use.
        steps: int, partly-required
            The number of steps to train the model.
        epochs: int, partly-required
            The number of epochs to train the model.
        train_feeds: dict(str, tf.placeholder), optional
            The model specific feeds for training, that have been
            defined in AbstractModel.fetch_feeds().
        valid_feeds: dict(str, tf.placeholder), optional
            The model specific feeds for validation, that have been
            defined in AbstractModel.fetch_feeds().
        on_validate: function or None, optional,
            A function with signature on_validate(runtime, gstep) that can
            be executed after each evaluation step.
        display_step: int, optional
            In which interval a averaged print-out of the current loss
            shoud be performed. Required for logging/testing only.
        summary_steps: int, optional
            In which interval we write a summary to TensorBoard.
        checkpoint_steps: int, optional
            In which interval we create a checkpoint.
        validation_steps: int, optional
            In which interval we perform a validation,
            which is also written to TensorBoard
        early_validation_at_step: int, optional
            On which (global) step we perform an early validation, basically that
            we get an early feedback of the losses, as well as if there is an error
            in doing a validation.
        do_checkpoints: Boolean, optioanl
            Whether we create checkpoints or not. Deactivate this for example programs
            where you do not want to fill up your disk.
        do_summary: Boolean, optional
            Whether we create summaries or not. Deactivate this for example programs
            where you do not want to fill up your disk.
        """
        assert not(steps <= 0 and epochs <= 0), "Either set 'steps' or 'epochs' parameter"
        assert not(steps > 0 and epochs > 0), "Not allowed to set both, 'steps' and 'epochs' parameter"
        
        with self.graph.as_default():
            batches_per_epoch = self.datasets.train.size // batch_size

            if epochs > 0:
                steps = batches_per_epoch * epochs

            self.datasets.train.reset()

            try:
                this_step = 0
                total_loss_sum = 0
                loss_sum = 0

                x_dummy = np.zeros([batch_size] + self.datasets.train.input_shape)
                y_dummy = np.zeros([batch_size] + self.datasets.train.target_shape)
                
                # add batch-size to summary. Copy is required to allow rerun training
                summaries_copy = copy.copy(self._summaries)
                summaries_copy.append(tf.scalar_summary('batch_size', batch_size))
                summary_op = tf.merge_summary(summaries_copy)

                while not self._coord.should_stop():
                    this_step += 1
                    if (this_step > steps):
                        break

                    if this_step % batches_per_epoch == 1:
                        epoch = (this_step - 1) // batches_per_epoch + 1
                        print("Starting epoch {}...".format(epoch))

                    start_time = time.time()

                    if isinstance(self.datasets.train, tt.datasets.base.AbstractQueueDataset):
                        batch_x = x_dummy
                        batch_y = y_dummy
                    else:
                        batch_x, batch_y = self.datasets.train.get_batch(batch_size)
                    feed = self._feed_func(batch_x, batch_y, batch_size, True)
                    feed.update({self._ph.input_from_queue: True \
                                 if isinstance(self.datasets.train, tt.datasets.base.AbstractQueueDataset) else False})
                    for key, value in train_feeds.iteritems():
                        feed.update({self._model_feeds[key]: value})
                    
                    if this_step == 1 and isinstance(self.datasets.train, tt.datasets.base.AbstractQueueDataset):
                        print("Filling queue with {} examples...".format(self.datasets.train.min_examples_in_queue))

                    # step counter is increment when train_op is executed
                    _, gstep, total_loss, loss = self.session.run([self._train_op,
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

                    if gstep % summary_steps == 0 or this_step == steps:
                        # summary
                        if do_summary == True:
                            summary_str = self.session.run(summary_op, feed_dict=feed)
                            self.summary_writer.add_summary(summary_str, gstep)
                            self.summary_writer.flush() 

                    if gstep == early_validation_at_step or this_step == steps or \
                        epochs == -1 and gstep % validation_steps == 0 or \
                        epochs > 0 and this_step % batches_per_epoch == 0:
                        # validate
                        print
                        self._test_internal(batch_size, self.datasets.valid, "validation", valid_feeds, do_summary)
                        print
                        
                        if on_validate is not None:
                            on_validate(self, gstep)
                            print

                    if do_checkpoints:
                        if gstep % checkpoint_steps == 0 or this_step == steps:
                            # save regular checkpoint
                            checkpoint_path = os.path.join(self.train_dir, "model.ckpt")
                            self._saver.save(self.session, checkpoint_path, global_step=self._global_step)

                        if epochs > 0:
                            if this_step % batches_per_epoch == 0:
                                # save epoch checkpoint
                                checkpoint_path = os.path.join(self.train_dir, "ep-{}_model.ckpt".format(epoch))
                                self._saver.save(self.session, checkpoint_path, global_step=self._global_step)

            except tf.errors.OutOfRangeError:
                print("Done training -- epoch limit reached")
    
    def predict(self, inputs, feeds={}):
        """Performs a prediction using the trained model.
        Parameters
        ----------
        inputs: numpy n-D array
            The inputs to the model to do the inference.
        feeds: dict(str, tf.placeholder), optional
            The model specific feeds, that have been
            defined in AbstractModel.fetch_feeds().
        Returns
        ---------
        The predictions of the model as an numpy n-D array.
        """
        with self.graph.as_default():
            batch_size = inputs.shape[0]
            y_dummy = np.zeros([batch_size] + self._ph.targets.get_shape().as_list()[1:])
            feed = self._feed_func(inputs, y_dummy, batch_size, False)
            feed.update({self._ph.input_from_queue: False})
            for key, value in feeds.iteritems():
                feed.update({self._model_feeds[key]: value})
            
            return self.session.run(self._inferences[0], feed_dict=feed)
        
    def validate(self, batch_size, feeds={}):
        """Performs a validation on the trained model using the validation
           dataset that was registered to the runtime.
        Parameters
        ----------
        batch_size: int
            The batch-size to use.
        feeds: dict(str, tf.placeholder), optional
            The model specific feeds, that have been
            defined in AbstractModel.fetch_feeds().
        """
        with self.graph.as_default():
            self._test_internal(batch_size, self.datasets.valid, "validation", feeds, False)
             
    def test(self, batch_size, feeds={}):
        """Performs a test on the trained model using the test
           dataset that was registered to the runtime.
        Parameters
        ----------
        batch_size: int
            The batch-size to use.
        feeds: dict(str, tf.placeholder), optional
            The model specific feeds, that have been
            defined in AbstractModel.fetch_feeds().
        """
        with self.graph.as_default():
            self._test_internal(batch_size, self.datasets.test, "test", feeds, False)
           
    def _test_internal(self, batch_size, dataset, title, feeds, do_summary):
        """Actually performs the validation/testing of the given dataset.
        Parameters
        ----------
        batch_size: int
            The batch size to use.
        dataset: AbstractDataset
            The dataset to use, typically either the validation or test set.
        title: str
            The title to use for the print-outs, basically just to see if we use
            the test or validation set.
        feeds: dict(str, tf.placeholder), optional
            The model specific feeds, that have been
            defined in AbstractModel.fetch_feeds().
        do_summary: Boolean
            Whether the validation results should be written to summary.
            Basically should be set to True while training only.
        """
        if dataset is None:
            print("No {} dataset registered. Skipping.".format(title))
            return
        
        batches_per_epoch = dataset.size // batch_size
        num_batches = batches_per_epoch
        
        # get current gstep from session
        gstep = self.gstep
        
        print("@{:6d}: Starting {} (batch-size: {}, dataset-size: {}):" \
              .format(gstep, title, batch_size, dataset.size))
        
        dataset.reset()
        loss_sum = 0
        x_dummy = np.zeros([batch_size] + dataset.input_shape)
        y_dummy = np.zeros([batch_size] + dataset.target_shape)
        progress = tt.utils.ui.ProgressBar(num_batches * batch_size)
        for b in xrange(num_batches):
            if isinstance(dataset, tt.datasets.base.AbstractQueueDataset):
                batch_x = x_dummy
                batch_y = y_dummy
            else:
                batch_x, batch_y = dataset.get_batch(batch_size)
            feed = self._feed_func(batch_x, batch_y, batch_size, False)
            feed.update({self._ph.input_from_queue: True \
                         if isinstance(dataset, tt.datasets.base.AbstractQueueDataset) else False})
            for key, value in feeds.iteritems():
                feed.update({self._model_feeds[key]: value})

            this_loss = self.session.run(self._loss, feed_dict=feed)
            loss_sum += this_loss
            progress.update((b+1) * batch_size, [('loss', this_loss)])
            
        if do_summary:
            avg_loss = loss_sum / num_batches
            loss_summary = tf.scalar_summary("{}_loss".format(title), avg_loss)
            summary_str = self.session.run(loss_summary)
            self.summary_writer.add_summary(summary_str, gstep)
            self.summary_writer.flush()
    
    def _create_session(self):
        """Creates the TensorFlow session."""
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self._gpu.memory_fraction,
            allow_growth=self._gpu.allow_growth)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
    def close(self):
        """Closes the runtime and releases the all threads."""
        self._coord.request_stop()
        print("Coordinator stopped.")

        # Wait for threads to finish
        if self._threads is not None:
            self._coord.join(self._threads)
        
        self.session.close()
        self._session = None
        
    @property
    def graph(self):
        """Gets the graph."""
        return self._graph

    @property
    def session(self):
        """Gets or creates the session."""
        if self._session is None:
            self._session = self._create_session()
        return self._session
    
    @property
    def datasets(self):
        """Gets the datasets as a named tuple.
           Use the members ds.train, ds.valid or ds.test
           of the returned tuple."""
        return self._datasets
    
    @property
    def gstep(self):
        """Gets the global step value as integer."""
        return self.session.run(self._global_step)
    
    @property
    def placeholders(self):
        """Gets the placeholders as a named tuple."""
        return self._ph

    @property
    def summary_writer(self):
        """Gets or creates the summary writer."""
        if self._summary_writer is None:
            self._summary_writer = tf.train.SummaryWriter(self.train_dir, self.session.graph)
        return self._summary_writer
    
    @property
    def train_dir(self):
        """Gets the training directory."""
        return self._train_dir
    
    @property
    def max_checkpoints_to_keep(self):
        """Gets the number of max. checkpoints to keep."""
        return self._max_checkpoints_to_keep
    
    
    
class DefaultRuntime(AbstractRuntime):
    """The default runtime, where TensorFlow itself
       descides where to assign the ops to."""
    
    def __init__(self, train_dir='train', max_checkpoints_to_keep=5,
                 gpu_allow_growth=True, gpu_memory_fraction=1.0):
        """Creates the default runtime instance.
        Parameters
        ----------
        train_dir: str, optional
            The training directory for checkpoints and summary files.
        max_checkpoints_to_keep: int, optional
            The number of last checkpoints to keep. Defaults to 5.
            Use 0 or None to keep all checkpoints.
        gpu_allow_growth: Boolean, optional
            Whether the GPUS is allowed to allocate memory dynamically.
            Has the advantage to only use that much memory as it really needs,
            but the downside of memory fragmentation and probably lower performance.
        gpu_memory_fraction: float in range (0, 1], optional
            The fraction of the (currently available) memory it is allows to reserve.
        """
        print("Launing default runtime...")
        
        device_list = tt.hardware.get_cuda_devices()
        if len(device_list) == 0:
            print("CPU-only mode or selecting GPU device: 0")
        else:
            print("Selecting GPU device: {}".format(device_list[0]))
        
        super(DefaultRuntime, self).__init__(train_dir, max_checkpoints_to_keep,
                                             gpu_allow_growth, gpu_memory_fraction)
        
    @tt.utils.attr.override
    def _build_computation_graph(self, x, y, opt):
        # Build inference Graph.This function constructs 
        # the entire model but shares the variables across all towers.
        with tf.name_scope("inference"):
            inference = self._model.inference(x, y,
                                              feeds=self._model_feeds,
                                              is_training=self._ph.is_training,
                                              device_scope=None, memory_device=None)
            
            # ensure the inference shape is fully defined and equal to target shape
            inference = tf.reshape(inference, [-1] + y.get_shape().as_list()[1:],
                                   name="ensure_shape")
            
            self._inferences.append(inference)
        
        with tf.name_scope("loss"):
            loss = self._model.loss(inference, y, device_scope=None)
            
        with tf.name_scope("total_loss"):
            total_loss = self._model.total_loss(loss, inference, y, device_scope=None)

        # Generate moving averages of all losses and associated summaries
        loss_averages_op = tt.board.loss_summary([total_loss, loss] + \
                                                 tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) + \
                                                 tf.get_collection(INTERMEDIATE_LOSSES),
                                                 decay=0.9)

        # Compute gradients
        with tf.control_dependencies([loss_averages_op]):
            grads = opt.compute_gradients(total_loss)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)    

        return grads, summaries, total_loss, loss
        


class MultiGpuRuntime(AbstractRuntime):
    """Advanced runtime that supports the use of multiple GPUs.
       All variables (have to be) stored on the CPU, and each batch
       is split according to the number of GPUs."""
    
    def __init__(self, num_gpus=2, train_dir='train', max_checkpoints_to_keep=5,
                 gpu_allow_growth=True, gpu_memory_fraction=1.0):
        """Creates a base runtime.
        Parameters
        ----------
        num_gpus: int, optioanl
            The number of GPUs to use.
        train_dir: str, optional
            The training directory for checkpoints and summary files.
        max_checkpoints_to_keep: int, optional
            The number of last checkpoints to keep. Defaults to 5.
            Use 0 or None to keep all checkpoints.
        gpu_allow_growth: Boolean, optional
            Whether the GPUS is allowed to allocate memory dynamically.
            Has the advantage to only use that much memory as it really needs,
            but the downside of memory fragmentation and probably lower performance.
        gpu_memory_fraction: float in range (0, 1], optional
            The fraction of the (currently available) memory it is allows to reserve.
        """
        print("Launing Multi-GPU runtime...")
        
        device_list = tt.hardware.get_cuda_devices()
        if len(device_list) == 0:
            print("Selecting all GPU devices.")
        else:
            assert len(device_list) >= num_gpus, "Not enough GPU devices available."
            print("Selecting GPU devices: {}".format(device_list[0:num_gpus]))
        self._num_gpus = num_gpus
        
        super(MultiGpuRuntime, self).__init__(train_dir, max_checkpoints_to_keep,
                                              gpu_allow_growth, gpu_memory_fraction)
        
    @tt.utils.attr.override
    def _build_computation_graph(self, x, y, opt):
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
                    with tf.name_scope("inference"):
                        inference = self._model.inference(this_inputs, this_targets,
                                                          feeds=self._model_feeds,
                                                          is_training=self._ph.is_training,
                                                          device_scope=scope,
                                                          memory_device='/cpu:0')
                        
                        # ensure the inference shape is fully defined and equal to target shape
                        inference = tf.reshape(inference, [-1] + this_targets.get_shape().as_list()[1:],
                                               name="ensure_shape")
                        
                        self._inferences.append(inference)
                    
                    with tf.name_scope("loss"):
                        loss = self._model.loss(inference, this_targets, device_scope=scope)
                    
                    with tf.name_scope("total_loss"):
                        total_loss = self._model.total_loss(loss, inference, this_targets, device_scope=scope)

                    # Calculate the moving averages of the loss for one tower of the model
                    loss_averages_op = tt.board.loss_summary([total_loss, loss] + \
                                                             tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) + \
                                                             tf.get_collection(INTERMEDIATE_LOSSES, scope),
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
        
        return grads, summaries, total_loss, loss
        
    @tt.utils.attr.override
    def train(self, batch_size, steps=-1, epochs=-1, feeds={}, on_validate=None,
              display_step=10, do_checkpoints=True, do_summary=True):
        assert batch_size % float(self.num_gpus) == 0, "Batch-size has to be multiples of 'num_gpus'."
        with tf.device('/cpu:0'):
            return super(MultiGpuRuntime, self).train(batch_size, steps, epochs, feeds, display_step,
                                                      do_checkpoints, do_summary)              
    
    @tt.utils.attr.override
    def validate(self, batch_size, feeds={}):
        assert batch_size % float(self.num_gpus) == 0, "Batch-size has to be multiples of 'num_gpus'."
        return super(MultiGpuRuntime, self).validate(batch_size, feeds)
    
    @tt.utils.attr.override
    def test(self, batch_size, feeds={}):
        assert batch_size % float(self.num_gpus) == 0, "Batch-size has to be multiples of 'num_gpus'."
        return super(MultiGpuRuntime, self).test(batch_size, feeds)
        
    
    @tt.utils.attr.override
    def predict(self, inputs, feeds={}):
        # Workaround: Let each tower do the same stuff, but only use the result of the
        #             first tower. Required to support e.g. batch-size 1 or odd batches.
        inputs_concat = inputs
        for i in xrange(self.num_gpus - 1):
            inputs_concat = np.concatenate((inputs_concat, inputs))
        
        return super(MultiGpuRuntime, self).predict(inputs_concat, feeds)
        
    @property
    def num_gpus(self):
        """Gets the number of used GPUs."""
        return self._num_gpus

    
    
def show_trainable_parameters(verbose=False):
    """Shows the number of trainable parameters in this graph."""
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        if verbose:
            print("{}: {}".format(variable.name, variable_parametes))
        total_parameters += variable_parametes
    if verbose:
        print("----------------------------------------")
    print("Total model-params: {}".format(total_parameters))