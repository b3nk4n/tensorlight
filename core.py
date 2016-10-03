from __future__ import print_function

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

CHECKPOINT_FILE = "model.ckpt"
MODEL_PARAMS_FILE = "model.json"
OPTIMIZER_PARAMS_FILE = "optimizer.json"

LATEST_CHECKPOINT = 'LATEST'
LOG_LOSSES = 'log_losses'


class AbstractRuntime(object):
    """Abstract runtime."""
    __metaclass__ = ABCMeta
    
    def __init__(self, train_dir, gpu_devices,
                 gpu_allow_growth=True, gpu_memory_fraction=1.0):
        """Creates a base runtime.
        Parameters
        ----------
        train_dir: str
            The training directory for checkpoints and summary files.
        gpu_devices: list(int)
            The list of the currently used GPU device IDs.
            Internally, the runtime filters the devices to select only these GPU devices,
            to prevent TensorFlow to allocate memory on other devices.
            If None or empty list, TensorFlow handles device assignment manually or
            we use a CPU only system. GPU masking will be cleared.
        gpu_allow_growth: Boolean, optional
            Whether the GPUS is allowed to allocate memory dynamically.
            Has the advantage to only use that much memory as it really needs,
            but the downside of memory fragmentation and probably lower performance.
        gpu_memory_fraction: float in range (0, 1], optional
            The fraction of the (currently available) memory it is allows to reserve.
        """
        assert gpu_memory_fraction > 0 and gpu_memory_fraction <= 1, "GPU memory fraction has to be in range (0,1]."
        
        self._graph = None
        self._session = None
        self._datasets = collections.namedtuple("datasets", ("train", "valid", "test"))
        self._model = None
        self._inferences = []
        
        # set Adam optimizer as default
        self._optimizer = tt.training.Optimizer(tt.training.ADAM, 0.001)

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
        
        # eval-dict might contain multiple items for 
        self._eval_dict = {}
        
        # placeholders and variables
        self._global_step = None
        self._ph = collections.namedtuple("placeholders", ("inputs",
                                                           "targets"
                                                           "is_training",
                                                           "batch_size",
                                                           "input_from_queue"))
        self._ph.is_training = None
        self._ph.batch_size = None
        self._ph.input_from_queue = None    
            
        self._train_dir = train_dir
        
        self._gpu = collections.namedtuple("gpu", ("devices",
                                                   "allow_growth",
                                                   "memory_fraction"))
        self._gpu.allow_growth = gpu_allow_growth
        self._gpu.memory_fraction = gpu_memory_fraction
        # mask the used devices to enuse not to use memory of other devices
        self._gpu.devices = gpu_devices
        tt.hardware.set_cuda_devices(gpu_devices)
           
    def register_datasets(self, train_ds=None, valid_ds=None, test_ds=None):
        """Registers the datasets.
           Note: You might want to register no dataset in case you only do predictions
                 on a pre-trained dataset.
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
        
    def unregister_datasets(self):
        """Unregisters all datasets.
           This can be used before re-building the model.
        """
        self.register_datasets(None, None, None)
        
    def register_model(self, model):
        """Registers the model.
        Parameters
        ----------
        model: AbstractModel
            The model to register.
        """
        if self._model is not None:
            raise ValueError("It is not allowed to change th model registration at runtime.")
        
        self._model = model
        
    def register_optimizer(self, optimizer):
        """Registers the optimizer.
           In case no optimizer was defined, Adam optimizer will be used.
        Parameters
        ----------
        optimizer: tt.training.Optimizer
            The optimizer to use while graph construction.
        """
        self._optimizer = optimizer
        

    def build(self, is_autoencoder=False, input_shape=None, target_shape=None,
              max_checkpoints_to_keep=5, track_ema_variables=True, restore_checkpoint=None,
              restore_ema_variables=False, restore_model_params=False, restore_optimizer_params=False,
              eval_mode=False, verbose=False):
        """ Builds the model. This must be calles before training, validation, testing or prediction.
            This method can be called a second time to re-create a model. In case the dataset's  the
            input shape or target shape, or the explicit input/target-shape changes, it required a model
            that is fully-convolutional.
        Parameters
        ----------
        is_autoencoder: Boolean, optional
            Whether we build an autoencoder, where the inputs equals the targets.
        input_shape: list(int), optional
            Spcifies the shape (excluding batch-size!) for the placeholder and the input to the model.
            Only used when no dataset is registered, because a dataset already defines the shape.
            This can be useful when to want to do predictions on a different shape that the model was trained.
        target_shape: list(int), optional
            Spcifies the shape (excluding batch-size!) for the placeholder and the target/output of the model,
            as well as the ground truth (GT) shape.
            Only used when no dataset is registered, because a dataset already defines the shape.
            This can be useful when to want to do predictions on a different shape that the model was trained.
        restore_checkpoint: str or int, optional
            The filename of the checkpoint file or step-number within 'train_dir' to restore.
            Use 'LATEST' or 'tt.core.LATEST_CHECKPOINT' to restore the lastest file.
        max_checkpoints_to_keep: int, optional
            The number of last checkpoints to keep. Defaults to 5.
            Use 0 or None to keep all checkpoints.
        track_ema_variables: Boolean, optional
            Indicates whether exponential moving averages of the trainable variables should be
            tracked (default) or not (False) during training.
            When training a model, it is often beneficial to maintain moving averages of
            the trained parameters.  Evaluations that use averaged parameters sometimes
            produce significantly better results than the final trained values.
            Tracking EMA variables can be disabled, but it is strongly recommended to keep it active.
        restore_ema_variables: Boolean, optional
            Indicates whether the original variable values should be restore (default) or its
            created exponential moving averages during training (True).
            It is only used when a model gets restored or recreated.
        restore_model_params: Boolean, optional
            Whether to restore the model parameters from the training directory. This will override
            all parameters of the registered model object.
        restore_optimizer_params: Boolean, optional
            Whether to restore the optimizer parameters from the training directory. 
            This will override all parameters of the registered optimizer object.
        eval_mode: Boolean, optional
            Flag that can be activated in evaluation mode in order to not restore EMA variables
            for all log-losses. These can lead the graph-construction to fail, even when these
            are not needed when doing predictions or evaluations.
        verbose: Boolean, optional
            Set to True to show additional construction/variable information.
        """
        assert self._model is not None, "Register a model first."
        
        # restore model params
        if restore_model_params:
            print("Restoring model parameters...")
            self._model.load(os.path.join(self.train_dir, MODEL_PARAMS_FILE))
            if verbose:
                print()
                self._model.print_params()
                print()
                    
        # restore optimizer params
        if restore_optimizer_params:
            print("Restoring optimizer parameters...")
            self._optimizer.load(os.path.join(self.train_dir, OPTIMIZER_PARAMS_FILE))
            if verbose:
                print()
                self._optimizer.print_params()
                print() 
        
        # take train set as reference for input and target shape
        reference_dataset = self.datasets.train
        if reference_dataset is None:
            # 1st fallback: validation set
            reference_dataset = self.datasets.valid
        if reference_dataset is None:
            # 2nd fallback: test set
            reference_dataset = self.datasets.valid
        
        if reference_dataset is None:
            # use explicit input/target shape defined for this build
            is_queue_dataset = None
            
            if input_shape == None or (not(is_autoencoder) and target_shape == None):
                raise ValueError("No dataset are registered, input and target shapes have to be defined explicitely.")
        else:
            # use the input/target shape defined by the dataset
            is_queue_dataset = isinstance(reference_dataset, tt.datasets.base.AbstractQueueDataset)
            input_shape = reference_dataset.input_shape
            target_shape = reference_dataset.target_shape
                
        recreate = False
        if self._graph is not None:
            recreate = True
            # re-create model
            with self.graph.as_default():
                # clear previous model-inferences
                del self._inferences[:]
                self._inferences = []
            
                if restore_checkpoint is None:
                    # use new saver to not modify 'max_to_keep' of global saver
                    saver = tf.train.Saver()
                    tmp_name = "/tmp/tmp-{}.ckpt".format(int(time.time()))
                    saver.save(self.session, tmp_name)
                
                 # close session to ensure all resources are released
                self.close()

        # crate a new graph
        self._graph = tf.Graph()

        with self.graph.as_default():
            # runtime placeholders and variables
            self._global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, trainable=False,
                                                initializer=tf.zeros_initializer)
            self._ph.is_training = tf.placeholder(tf.bool, name='is_training')
            self._ph.batch_size = tf.placeholder(tf.int32, name='batch_size')
            self._ph.input_from_queue = tf.placeholder(tf.bool, name='input_from_queue')
            
            # input placeholders
            self._ph.inputs = tf.placeholder(tf.float32, [None] + input_shape, "X")
            if is_autoencoder:
                self._ph.targets = tf.placeholder(tf.float32, [None] + input_shape, "Y")
            else:
                self._ph.targets = tf.placeholder(tf.float32, [None] + target_shape, "Y")

            if is_queue_dataset:
                with tf.device("/cpu:0"):
                    # doing inputs on CPU is generally a good idea
                    inputs, targets = reference_dataset.get_batch(self._ph.batch_size)
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
            
            # build the optimizer instance
            with tf.name_scope('optimizer'):
                opt, lr = self.optimizer.build(self._global_step)
              
            # install the model and make global variables availalbe
            self._model.install(self._global_step)
            
            # build (multi-)device specific computation graph for inference
            grads, summaries, total_loss, loss, eval_dict = self._build_computation_graph(x, y, opt, eval_mode)
            
            # Apply gradients
            apply_gradient_op = opt.apply_gradients(grads, global_step=self._global_step)

            # Add summaries
            summaries.append(tf.scalar_summary('learning_rate', lr))
            summaries.extend(tt.board.gradients_histogram_summary(grads))
            summaries.extend(tt.board.variables_histogram_summary())

            # Track the moving averages of all trainable variables
            variable_averages = tf.train.ExponentialMovingAverage(0.9999, self._global_step)

            if track_ema_variables:
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                train_op = tf.group(apply_gradient_op, variables_averages_op, name="train_op")
            else:
                # exclude the 
                train_op = tf.group(apply_gradient_op, name="train_op")
            
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
            self._eval_dict = eval_dict
                
            # Create a saver to store checkpoints of the model
            if restore_ema_variables:
                restore_vars = tf.all_variables()
            else:
                restore_vars = variable_averages.variables_to_restore()  
            
            self._saver = tf.train.Saver(var_list=restore_vars,
                                         max_to_keep=max_checkpoints_to_keep)
            
            def perform_restore(saver, filepath, restore_ema):
                """Restores a checkpoint, either the raw variables or the EMA values."""
                if restore_ema:
                    # Init all at first to ensure no non-initialized variables.
                    # This is just a workaround and might be buggy.
                    # It does not guarantee to restore all EMA vars.
                    init_op = tf.initialize_all_variables()
                    self.session.run(init_op)
                    try:
                        print("Restoring EMA variables...")
                        saver.restore(self.session, filepath)
                    except tf.errors.NotFoundError:
                        print("Warning: Could not restore model, because no EMA variables have been found.",\
                              "Use 'restore_ema_variables=False' instead.")
                else:
                    print("Restoring variables...")
                    saver.restore(self.session, filepath)
                return saver
                
            if restore_checkpoint is None:
                if recreate:
                    saver = tf.train.Saver(var_list=restore_vars)
                    perform_restore(saver, tmp_name, restore_ema_variables)
                else:
                    # start session and init all variables
                    print("Initializing variables...")
                    init_op = tf.initialize_all_variables()
                    self.session.run(init_op)
            else:
                if restore_checkpoint == LATEST_CHECKPOINT:
                    checkpoint_path = tf.train.latest_checkpoint(self.train_dir)
                    assert checkpoint_path is not None, "No latest checkpoint file found."
                elif isinstance(restore_checkpoint, int):
                    checkpoint_path = os.path.join(self.train_dir,
                                                   "{}-{}".format(CHECKPOINT_FILE, restore_checkpoint))
                else:
                    checkpoint_path = os.path.join(self.train_dir, restore_checkpoint)
                    
                print("Selected checkpoint file: {}".format(checkpoint_path))
                perform_restore(self._saver, checkpoint_path, restore_ema_variables)

            # creates coordinatior and queue threads
            self._coord = tf.train.Coordinator()
            if is_queue_dataset:
                self._threads = tf.train.start_queue_runners(sess=self.session, coord=self._coord)
                
            # Model information
            print()
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
        A tuple of (grads, summaries, total_loss, loss, eval_dict)
        """
        pass
        
    def train(self, batch_size, valid_batch_size=None, steps=-1, epochs=-1, train_feeds={}, valid_feeds={},
              on_validate=None, display_steps=25, summary_steps=100, checkpoint_steps=1000,
              validation_steps=1000, extra_validations=[100, 250, 500, 750],
              do_checkpoints=True, do_summary=True, save_model_params=True, save_optimizer_params=True):
        """Train the model.
           Note that either 'steps' or 'epochs' has to be defined as a param.
        Parameters
        ----------
        batch_size: int
            The batch size to use for training (and for validation, in case
            'valid_batch_size') is not defined.
        valid_batch_size: int or None, optional
            The batch size to use for validation, or None to use the same as for training.
            You might want to use a different batch_size for validation, to make sure that
            every example is actually evaluated.
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
        display_steps: int, optional
            In which interval a averaged print-out of the current loss
            shoud be performed. Required for logging/testing only.
        summary_steps: int, optional
            In which interval we write a summary to TensorBoard.
        checkpoint_steps: int, optional
            In which interval we create a checkpoint.
        validation_steps: int, optional
            In which interval we perform a validation,
            which is also written to TensorBoard
        extra_validations: list(int) or None, optional
            On which (global) step we perform extra validations, basically that we
            can get an early feedback of the losses, as well as if there is an error
            in doing a validation. Or when there are huge jumps at the beginning of
            the training.
        do_checkpoints: Boolean, optioanl
            Whether we create checkpoints or not. Deactivate this for example programs
            where you do not want to fill up your disk.
        do_summary: Boolean, optional
            Whether we create summaries or not. Deactivate this for example programs
            where you do not want to fill up your disk.
        save_model_params: Boolean, optional
            Whether to save the model params in the training directory or not.
            This will override an existing file in case of re-training.
        save_optimizer_params: Boolean, optional
            Whether to save the optimizer params in the training directory or not.
            This will override an existing file in case of re-training.
        """
        assert not(steps <= 0 and epochs <= 0), "Either set 'steps' or 'epochs' parameter"
        assert not(steps > 0 and epochs > 0), "Not allowed to set both, 'steps' and 'epochs' parameter"
        
        assert batch_size % self.num_computing_devices == 0, \
            "Batch-size has to be a multiple of computing devices used."
        assert valid_batch_size is None or valid_batch_size % self.num_computing_devices == 0, \
            "Validation batch-size has to be a multiple of computing devices used."
        
        dataset = self.datasets.train
        if not self._check_dataset_registered(dataset):
            return
        
        # save parameters as JSON
        if save_model_params:
            self._model.save(os.path.join(self.train_dir, MODEL_PARAMS_FILE))
        if save_optimizer_params:
            self._optimizer.save(os.path.join(self.train_dir, OPTIMIZER_PARAMS_FILE))
        
        if valid_batch_size is None:
            # take training batch_size as fallback.
            valid_batch_size = batch_size
            
        batches_per_epoch = dataset.size // batch_size

        if epochs > 0:
            steps = batches_per_epoch * epochs

        dataset.reset()
        
        with self.graph.as_default():
            # take the CPU as root device in case of many GPUs
            device_scope = None if self.num_computing_devices == 1 else '/cpu:0'
            with tf.device(device_scope):
                try:
                    this_step = 0
                    step_divisor = 0
                    total_loss_sum = 0
                    loss_sum = 0

                    x_dummy = np.zeros([batch_size] + dataset.input_shape, np.float32)
                    y_dummy = np.zeros([batch_size] + dataset.target_shape, np.float32)

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

                        # prepare feeding
                        if isinstance(dataset, tt.datasets.base.AbstractQueueDataset):
                            batch_x = x_dummy
                            batch_y = y_dummy
                        else:
                            batch_x, batch_y = dataset.get_batch(batch_size)
                        feed = self._feed_func(batch_x, batch_y, batch_size, True)
                        feed.update({self._ph.input_from_queue: True \
                                     if isinstance(dataset, tt.datasets.base.AbstractQueueDataset) else False})
                        for key, value in train_feeds.iteritems():
                            feed.update({self._model_feeds[key]: value})

                        if this_step == 1 and isinstance(dataset, tt.datasets.base.AbstractQueueDataset):
                            print("Filling queue with {} examples...".format(dataset.min_examples_in_queue))

                        # step counter is increment when train_op is executed
                        _, gstep, total_loss, loss = self.session.run([self._train_op,
                                                                      self._global_step,
                                                                      self._total_loss,
                                                                      self._loss],
                                                                     feed_dict=feed)
                        duration = time.time() - start_time

                        assert not np.isnan(loss), 'Warning: Model diverged with loss = NaN'

                        step_divisor += 1
                        total_loss_sum += total_loss
                        loss_sum += loss
                        if this_step == 1 or gstep % display_steps == 0:
                            # info
                            num_examples_per_step = batch_size
                            examples_per_sec = num_examples_per_step / duration
                            sec_per_batch = float(duration)
                            avg_total_loss = total_loss_sum / step_divisor
                            avg_loss = loss_sum / step_divisor
                            step_divisor = 0
                            total_loss_sum = 0
                            loss_sum = 0
                            print("@{:5d}: loss: {:7.3f}, t-loss: {:7.3f} ({:7.1f}" \
                                  " examples/sec, {:5.2f} sec/batch)" \
                                  .format(gstep, avg_loss, avg_total_loss,
                                          examples_per_sec, sec_per_batch))

                        if gstep % summary_steps == 0 or this_step == steps:
                            # summary
                            if do_summary == True:
                                summary_str = self.session.run(summary_op, feed_dict=feed)
                                self.summary_writer.add_summary(summary_str, gstep)
                                self.summary_writer.flush() 

                        if gstep in extra_validations or this_step == steps or \
                            epochs == -1 and gstep % validation_steps == 0 or \
                            epochs > 0 and this_step % batches_per_epoch == 0:
                            # validate
                            print()
                            self._test_internal(valid_batch_size, self.datasets.valid,
                                                "validation", valid_feeds, do_summary)
                            print()

                            if on_validate is not None:
                                on_validate(self, gstep)
                                print()

                        if do_checkpoints:
                            if gstep % checkpoint_steps == 0 or this_step == steps or \
                                epochs > 0 and this_step % batches_per_epoch == 0:
                                # save regular checkpoint
                                checkpoint_path = os.path.join(self.train_dir, CHECKPOINT_FILE)
                                self._saver.save(self.session, checkpoint_path,
                                                 global_step=self._global_step)

                except tf.errors.OutOfRangeError:
                    print("Interrupted: Queue runners are out of range. Epoch limit reached?")
    
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
            
            # prepare feeding
            y_dummy = np.zeros([batch_size] + self._ph.targets.get_shape().as_list()[1:], np.float32)
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
        assert batch_size % self.num_computing_devices == 0, \
            "Batch-size has to be a multiple of computing devices used."
        
        if not self._check_dataset_registered(dataset):
            return
        
        batches_per_epoch = dataset.size // batch_size
        num_batches = batches_per_epoch
        
        # get current gstep from session
        gstep = self.gstep
        
        # ops to execute on validation
        eval_names = ["loss"]
        eval_ops = [self._loss]
        for name, eval_op in self._eval_dict.iteritems():
            eval_names.append(name.lower()) 
            eval_ops.append(eval_op)
        
        print("@{:6d}: Starting {} (batch-size: {}, dataset-size: {}):" \
              .format(gstep, title, batch_size, dataset.size))
        
        dataset.reset()
        eval_sums = np.zeros(len(eval_ops))
        x_dummy = np.zeros([batch_size] + dataset.input_shape, np.float32)
        y_dummy = np.zeros([batch_size] + dataset.target_shape, np.float32)
        progress = tt.utils.ui.ProgressBar(num_batches * batch_size)
        for b in xrange(num_batches):
            # prepare feeding
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

            # run evaluation for all ops
            this_evals_tuple = self.session.run(eval_ops, feed_dict=feed)
            this_evals = np.array(this_evals_tuple)
            eval_sums += this_evals
            
            # create status list for progress bar
            status_list = []
            for i, name in enumerate(eval_names):
                status_list.append((name, this_evals[i]))
            
            progress.update((b+1) * batch_size, status_list)
            
        if do_summary:
            avg_evals = eval_sums / num_batches
            
            # execute all summaries in a single run
            eval_summaries = []
            for i, name in enumerate(eval_names):
                eval_summaries.append(tf.scalar_summary("{}_{}".format(title, name), avg_evals[i]))
            summary_strings = self.session.run(eval_summaries)
            
            # add to summary writer
            for sum_str in summary_strings:
                self.summary_writer.add_summary(sum_str, gstep)
            self.summary_writer.flush()
            
    def _check_dataset_registered(self, dataset):
        """Checks if the corresponding dataset has been registered."""
        if dataset is None:
            print("No proper dataset registered. Skipping.")
            return False
        return True
    
    def _create_session(self):
        """Creates the TensorFlow session."""
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self._gpu.memory_fraction,
            allow_growth=self._gpu.allow_growth)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
    def close(self):
        """Closes the runtime and releases the all threads."""
        if self._coord is not None:
            self._coord.request_stop()
        
        if self._threads is not None:
            # Wait for threads to finish
            self._coord.join(self._threads)
        
        self.session.close()
        self._session = None
        
    def list_params(self, verbose=False):
        """Lists all internal parameters."""
        print()
        self._model.print_params()
        print()
        self._optimizer.print_params()
        print()
        if self.graph is not None:
            with self.graph.as_default():
                tt.core.show_trainable_parameters(verbose)
            print()
         
    def list_checkpoints(self):
        """Lists all checkpoint file of the used directory"""
        checkpointslike = tt.utils.path.get_filenames(self.train_dir, "*.ckpt*")
        checkpoints = [cp for cp in checkpointslike if not cp.endswith(".meta")]
        checkpoints.sort()
        return checkpoints
        
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
    def optimizer(self):
        """Gets the optimizer. Changes at runtime do not inflence the optimizer,
           since it is created during graph construction. Use it read-only."""
        return self._optimizer
    
    @property
    def gstep(self):
        """Gets the global step value as integer."""
        return self.session.run(self._global_step)
    
    @property
    def placeholders(self):
        """Gets the placeholders as a named tuple."""
        return self._ph
    
    @property
    def gpu(self):
        """Gets the gpu config as a named tuple."""
        return self._gpu
    
    @property
    def num_computing_devices(self):
        """Gets the total number of used computing devices."""
        if self.gpu.devices is None or len(self.gpu.devices) == 0:
            return 1
        return len(self.gpu.devices)

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
    
    
class DefaultRuntime(AbstractRuntime):
    """The default runtime, where TensorFlow itself
       descides where to assign the ops to."""
    
    def __init__(self, train_dir, gpu_devices=None,
                 gpu_allow_growth=True, gpu_memory_fraction=1.0):
        """Creates the default runtime instance.
        Parameters
        ----------
        train_dir: str
            The training directory for checkpoints and summary files.
        gpu_devices: list(int) or None, optional
            The list of the currently used GPU device IDs.
            Internally, the runtime filters the devices to select only these GPU devices,
            to prevent TensorFlow to allocate memory on other devices.
            If None or empty list, TensorFlow handles device assignment manually or
            we use a CPU only system.
        gpu_allow_growth: Boolean, optional
            Whether the GPUS is allowed to allocate memory dynamically.
            Has the advantage to only use that much memory as it really needs,
            but the downside of memory fragmentation and probably lower performance.
        gpu_memory_fraction: float in range (0, 1], optional
            The fraction of the (currently available) memory it is allows to reserve.
        """
        assert gpu_devices is None or isinstance(gpu_devices, list), "Define a valid device selection."
        
        if gpu_devices is not None and len(gpu_devices) > 1:
            # only select the first one
            gpu_devices = gpu_devices[:1]
        
        super(DefaultRuntime, self).__init__(train_dir, gpu_devices,
                                             gpu_allow_growth, gpu_memory_fraction)
        
    @tt.utils.attr.override
    def _build_computation_graph(self, x, y, opt, eval_mode):
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
            
        with tf.name_scope("evaluation"):
            eval_dict = self._model.evaluation(inference, y, device_scope=None)

        # Generate moving averages of all losses and associated summaries
        loss_averages_op = tt.board.loss_summary([total_loss, loss] + \
                                                 tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) + \
                                                 tf.get_collection(LOG_LOSSES) if not eval_mode else [],
                                                 decay=0.9)

        # Compute gradients
        with tf.control_dependencies([loss_averages_op]):
            grads = opt.compute_gradients(total_loss)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)    

        return grads, summaries, total_loss, loss, eval_dict
        


class MultiGpuRuntime(AbstractRuntime):
    """Advanced runtime that supports the use of multiple GPUs.
       All variables (have to be) stored on the CPU, and each batch
       is split according to the number of GPUs.
       
       Note: This runtime implementatin is currently still experimental.
             There might be init-error on some ops (LSTM-cells) or
             device-assignment errors (batch_norm). It is currently
             guaranteed works on simple models only.
       """
    def __init__(self, train_dir, gpu_devices=[0,1],
                 gpu_allow_growth=True, gpu_memory_fraction=1.0):
        """Creates a base runtime.
        Parameters
        ----------
        train_dir: str
            The training directory for checkpoints and summary files.
        gpu_devices: list(int)
            The list of the currently used GPU device IDs with length >= 2.
            Internally, the runtime filters the devices to select only these GPU devices,
            to prevent TensorFlow to allocate memory on other devices.
        gpu_allow_growth: Boolean, optional
            Whether the GPUS is allowed to allocate memory dynamically.
            Has the advantage to only use that much memory as it really needs,
            but the downside of memory fragmentation and probably lower performance.
        gpu_memory_fraction: float in range (0, 1], optional
            The fraction of the (currently available) memory it is allows to reserve.
        """
        assert gpu_devices is None or (isinstance(gpu_devices, list) and len(gpu_devices) > 1), \
            "Define a valid device selection."
        
        super(MultiGpuRuntime, self).__init__(train_dir, gpu_devices,
                                              gpu_allow_growth, gpu_memory_fraction)
        
    @tt.utils.attr.override
    def _build_computation_graph(self, x, y, opt, eval_mode):
        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_losses = []
        tower_total_losses = []
        eval_dicts = []
        splitted_x = tf.split(0, self.num_computing_devices, x)
        splitted_y = tf.split(0, self.num_computing_devices, y)
        for i in xrange(self.num_computing_devices):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    this_inputs = splitted_x[i]
                    this_targets = splitted_y[i]

                    # Build inference Graph.This function constructs 
                    # the entire model but shares the variables across all towers.
                    with tf.name_scope("inference"):
                        inference = self._model.inference(this_inputs, this_targets,
                                                          feeds=self._model_feeds,
                                                          is_training=self._ph.is_training,
                                                          device_scope=scope,
                                                          memory_device=None) # '/cpu:0'
                        # FIXME: inference(..., memory_device=...) should be set to '/cpu:0' according to
                        #        the TensorFlow CIFAR-10 example. But this causes init-errors on the LSMT-cells.
                        #        Doing no assignment to CPU-memory works. It might require more memory, but the
                        #        performance is almost the same.
                        
                        # ensure the inference shape is fully defined and equal to target shape
                        inference = tf.reshape(inference, [-1] + this_targets.get_shape().as_list()[1:],
                                               name="ensure_shape")
                        
                        self._inferences.append(inference)
                    
                    with tf.name_scope("loss"):
                        loss = self._model.loss(inference, this_targets, device_scope=scope)
                    
                    with tf.name_scope("total_loss"):
                        total_loss = self._model.total_loss(loss, inference, this_targets, device_scope=scope)
                        
                    with tf.name_scope("evaluation"):
                        eval_dict = self._model.evaluation(inference, this_targets, device_scope=scope)

                    # Calculate the moving averages of the loss for one tower of the model
                    loss_averages_op = tt.board.loss_summary([total_loss, loss] + \
                                                             tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) + \
                                                             tf.get_collection(LOG_LOSSES, scope) if not eval_mode else [],
                                                             decay=0.9)

                    with tf.control_dependencies([loss_averages_op]):
                        # only add total_loss there because this one is used on training
                        this_total_loss = tf.identity(total_loss)
                       
                    tower_losses.append(loss)
                    tower_total_losses.append(this_total_loss)
                        
                    eval_dicts.append(eval_dict)

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
        
        # average the losses for evaluation over all towers
        avg_loss = tf.reduce_mean(tf.pack(tower_losses))
        avg_total_loss = tf.reduce_mean(tf.pack(tower_total_losses))
        
        # average the evaluation dicts
        avg_eval_dict = {}
        for key in eval_dicts[0]:
            packed_scalar_ops = tf.pack([eval_dicts[i][key] for i in xrange(self.num_computing_devices)])
            avg_scalar_ops = tf.reduce_mean(packed_scalar_ops)
            avg_eval_dict.update({key: avg_scalar_ops})
            
        return grads, summaries, avg_total_loss, avg_loss, avg_eval_dict
        
    @tt.utils.attr.override
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
        # Workaround: Let each tower do the same stuff, but only use the result of the
        #             first tower. Required to support e.g. batch-size 1 or odd batches.
        inputs_concat = inputs
        for i in xrange(self.num_computing_devices - 1):
            inputs_concat = np.concatenate((inputs_concat, inputs))
        
        return super(MultiGpuRuntime, self).predict(inputs_concat, feeds)

    
def show_trainable_parameters(verbose=False):
    """Shows the number of trainable parameters in this graph.
    Parameters
    ----------
    verbose: Boolean, optional
        Show additional information and list the number of trainable
        variables per variable, not just the total sum.
    """
    total_width = 80
    trainable_vars = tf.trainable_variables()
    
    if len(trainable_vars) == 0:
        print("No model-params found.")
        return
    
    if verbose:
        print("-" * total_width)
    
    total_parameters = 0
    groups = {}
    for var in trainable_vars:
        # shape is an array of tf.Dimension
        shape = var.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim.value
        if verbose:
            print("{:69} | {:8d}".format(var.name, var_params))
        
        total_parameters += var_params
        
        group_name = var.name.split('/')[0]
        if group_name in groups:
            groups[group_name] += var_params
        else:
            groups.update({group_name: var_params})
    
    print("-" * total_width)
    for group, count in groups.iteritems():
        print("{:69} | {:8d}".format(group, count))
    print("=" * total_width)
    print("{:69} | {:8d}".format("TOTAL", total_parameters))
    print("-" * total_width)

    
def uninitialized_variables(session, var_list=None):
    """Gets the list of uninitialized variables.
       Note: this has to be evaluated on a session.
    Parameters
    ----------
    session: tf.Session
        The TensorFlow session to scan for uninitialized variables
    var_list: list(tf.Varaible) or None
        The list of variables to filter for uninitialized ones.
        Defaults to tf.all_variables() is used.
    """
    if var_list is None:
        var_list = tf.all_variables()

    reported_var_names = session.run(tf.report_uninitialized_variables(var_list))
    uninit_vars = []
    for name in reported_var_names:
        try:
            uninit_vars.append(tf.get_variable(name))
        except ValueError:
            print("Failed to collect variable {}. Skipping.", name)
            
    return uninit_vars
    
    
def initialize_uninitialized_variables(session, var_list=None):
    """Initializes all uninitialized variables.
    Parameters
    ----------
    session: tf.Session
        The TensorFlow session to scan for uninitialized variables
    var_list: list(tf.Varaible) or None
        The list of variables to filter for uninitialized ones.
        Defaults to tf.all_variables() is used.
    """
    uninit_vars = uninitialized_variables(session, var_list)
    session.run(tf.initialize_variables(uninit_vars))