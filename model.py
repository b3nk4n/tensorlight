import tensorflow as tf
import tensortools as tt
from abc import ABCMeta


class AbstractModel(object):
    """Abstract class as a template for a TensorFlow model.
       
       References: Inspired by http://danijar.com/structuring-your-tensorflow-models/
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, inputs, targets, reg_lambda=0.0, is_training=True,
                 device_scope=None, memory_device=None):
        """Creates the model instance that is shared accross all models.
        Parameters
        ----------
        inputs: 5D-Tensor of shape [batch_size, nstep, h, w, c]
            The inputs of the model.
        targets: 5D-Tensor of shape [batch_size, nstep, h, w, c]
            The target outputs of the model.
        reg_lambda: float, optional
            The regularization factor (lambda).
        is_training: Boolean, optional
            Flag inidcating traing or eval mode. E.g. used for batch norm.
        device_scope: str or None, optional
            The tower name in case of multi-GPU runs.
        memory_device: str, optional
            The device where there model should put it's variables,
            in case of multi-GPU runs.
        """
        self._inputs = inputs
        self._targets = targets
        self._reg_lambda = reg_lambda
        self._is_training = is_training
        self._device_scope = device_scope
        self._memory_device = memory_device
        
        # ensure graph is builded before TF inits all variables
        self.predictions
        self.loss
        self.total_loss
    
    @tt.utils.attr.lazy_abstractproperty
    def predictions(self):
        """Builds the graph (in case it is not already done) for inference.
           Implemented as a lazy property.
        Returns
        ----------
        Returns the predictions of the model.
        """
        pass  # abstract
    
    @tt.utils.attr.lazy_abstractproperty
    def loss(self):
        """Gets the loss of the model (excluding the regularization losses).
           Implemented as a lazy property.
        Returns
        ----------
        Returns the loss as a float.
        """
        pass  # abstract
      
    @tt.utils.attr.lazy_property
    def total_loss(self):
        """Gets the total loss of the model including the regularization losses.
           Implemented as a lazy property.
        Returns
        ----------
        Returns the total oss as a float.
        """
        wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(wd_losses) > 0:
            reg_loss = tf.add_n(wd_losses, name="reg_loss")
            total_loss = tf.add(self.loss, reg_loss, name="total_loss")
        else:
            total_loss = tf.identity(self.loss, name="total_loss")
        
        return total_loss

    @property
    def input_shape(self):
        """Gets the static input shape."""
        return self._inputs.get_shape().as_list()
    
    @property
    def output_shape(self):
        """Gets the static output shape."""
        return self._targets.get_shape().as_list()
    
    @property
    def batch_size(self):
        """Gets the dynamic shape of the batch size."""
        return tf.shape(self._inputs)[0]
    
    @property
    def inputs(self):
        """Gets the inputs tensor."""
        return self._inputs
    
    @property
    def targets(self):
        """Gets the targets tensor."""
        return self._targets
    
    @property
    def reg_lambda(self):
        """Gets the regularization factor (lambda) for weight decay."""
        return self._reg_lambda
    
    @property
    def is_training(self):
        """Gets the flag indicating whether we are in training or eval."""
        return self._is_training
    
    @property
    def device_scope(self):
        """Gets the device scope, used for multi-tower runtimes."""
        return self._device_scope
    
    @property
    def memory_device(self):
        """Gets the device where to store variables, used in multi-tower
           runtimes. None means no scrict assignment."""
        return self._memory_device