import os
import types
import jsonpickle

import tensorflow as tf
import tensortools as tt
from abc import ABCMeta, abstractmethod


class AbstractModel(object):
    """Abstract class as a template for a TensorFlow model.
       
       References: Inspired by http://danijar.com/structuring-your-tensorflow-models/
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, weight_decay=0.0):
        """Creates the base model instance that is shared accross all models.
           It allows to build multiple models using the same construction plan.
        Parameters
        ----------
        weight_decay: float, optional
            The weight decay regularization factor (lambda).
        """
        self._weight_decay = weight_decay
        self._global_step = None
        
    def fetch_feeds(self):
        """Can be overridden to fetch the model specific feeding dict.
           Do not call this method within the model. It is calles by the runtime
           to ensure the placeholder are created within the right graph.
        Returns
        ----------
        Returns a dictionary with values {str-key: tf.placeholder, ...}.
        """
        return {}
    
    def install(self, global_step):
        """Installs global variables from the runtime.
        Parameters
        ----------
        global_step: int-Tensor
            The global variable representing the global step. This might be useful
            for the model in case it uses internal decay variables.
        """
        self._global_step = global_step
        
    @abstractmethod
    def inference(self, inputs, targets, feeds={}, is_training=True,
                  device_scope=None, memory_device=None):
        """Builds the models inference.
           Note: Everytime this method function is called, a new
                 model instance in created.
        Parameters
        ----------
        inputs: 5D-Tensor of shape [batch_size, nstep, h, w, c]
            The inputs of the model.
        targets: 5D-Tensor of shape [batch_size, nstep, h, w, c]
            The target outputs of the model.
        feeds: dict(str, tf.placeholder), optional
            The model specific feeds, that have been
            defined in AbstractModel.fetch_feeds().
        is_training: Boolean, optional
            Flag inidcating training or eval mode. E.g. used for batch norm.
        device_scope: str or None, optional
            The tower name in case of multi-GPU runs.
        memory_device: str, optional
            The device where there model should put it's variables,
            in case of multi-GPU runs.
        """
        pass
    
    @abstractmethod
    def loss(self, predictions, targets, device_scope=None):
        """Gets the loss of the model.
        Parameters
        ----------
        predictions: n-D Tensor
            The predictions of the model.
        targets: n-D Tensor
            The targets/labels.
        device_scope: str or None, optional
            The tower name in case of multi-GPU runs.
        Returns
        ----------
        Returns the loss as a float.
        """
        pass
    
    def total_loss(self, loss, predictions, targets, device_scope=None):
        """Gets the total loss of the model including the regularization losses.
           Implemented as a lazy property.
        Parameters
        ----------
        loss: float32 Tensor
            The result of loss(predictions, targets) that should be included in this loss.
        predictions: n-D Tensor
            The predictions of the model.
        targets: n-D Tensor
            The targets/labels.
        device_scope: str or None, optional
            The tower name in case of multi-GPU runs.
        Returns
        ----------
        Returns the total loss as a float.
        """
        wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(wd_losses) > 0:
            reg_loss = tf.add_n(wd_losses, name="reg_loss")
            return tf.add(loss, reg_loss, name="loss_with_reg")
        else:
            # we have to wrap this with identity, because in in other case it
            # would rise a summary-writer error that this loss was already added
            return tf.identity(loss, name="loss_with_reg")
        
    def evaluation(self, predictions, targets, device_scope=None):
        """Returns a dict of {title: scalar-Tensor, ...} that are evaluated during 
           validation and training.
           Note:
               All names must be a valid filename, as they are used in TensorBoard. 
        predictions: n-D Tensor
            The predictions of the model.
        targets: n-D Tensor
            The targets/labels.
        device_scope: str or None, optional
            The tower name in case of multi-GPU runs.
        Returns
        ----------
        A dict of {title: scalar-Tensor, ...} to be executed in validation and testing.
        """
        return {}
    
    def save(self, filepath):
        """Saves the model parameters to the specifiec path as JSON.
        Parameters
        ----------
        filepath: str
            The file path.
        """
        # check and create dirs
        if not os.path.exists(os.path.dirname(filepath)):
            subdirs = os.path.dirname(filepath)
            if subdirs is not None and subdirs != '':
                os.makedirs(subdirs)
        
        with open(filepath, 'wb') as f:
            json = jsonpickle.encode(self)
            f.write(json)
            
    def load(self, filepath):
        """Load the model parameters from the specifiec path as JSON.
        Parameters
        ----------
        filepath: str
            The file path.
        """
        with open(filepath, 'r') as f:
            json = f.read()
            model = jsonpickle.decode(json)
            self.__dict__.update(model.__dict__)
    
    @tt.utils.attr.override
    def __getstate__(self):
        """Overridden for jsonpickle to exclude globa step."""
        state = self.__dict__.copy()
        del state['_global_step']
        return state
    
    def print_params(self):
        """Shows the model parameters."""
        params = self.__getstate__()
        
        def trim_prefix(text, prefix):
            # trim underscore prefix
            return text[text.startswith(prefix) and len(prefix):]
        
        def to_string(value):
            # <function sigmoid at 0x7f78b31bc410>
            if isinstance(value, types.FunctionType):
                return value.__name__.upper()
            return value

        print(">>> Model:")
        for name, value in params.iteritems():
            print("{:16}  ->  {}".format(trim_prefix(name, '_'), to_string(value)))

    @property
    def batch_size(self):
        """Gets the dynamic shape of the batch size."""
        return tf.shape(self._inputs)[0]
    
    @property
    def weight_decay(self):
        """Gets the regularization factor (lambda) for weight decay."""
        return self._weight_decay
    
    @property
    def global_step(self):
        """Gets the global step variable as a Tensor."""
        return self._global_step