import tensorflow as tf
import tensortools as tt
from abc import ABCMeta, abstractmethod


class AbstractModel(object):
    """Abstract class as a template for a TensorFlow model.
       
       References: Inspired by http://danijar.com/structuring-your-tensorflow-models/
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, reg_lambda=0.0):
        """Creates the base model instance that is shared accross all models.
           It allows to build multiple models using the same construction plan.
        Parameters
        ----------
        reg_lambda: float, optional
            The regularization factor (lambda).
        """
        self._reg_lambda = reg_lambda
        
    def fetch_feeds(self):
        """Can be overridden to fetch the model specific feeding dict.
           Do not call this method within the model. It is calles by the runtime
           to ensure the placeholder are created within the right graph.
        Returns
        ----------
        Returns a dictionary with values {str-key: tf.placeholder, ...}.
        """
        return {}
        
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

    @property
    def batch_size(self):
        """Gets the dynamic shape of the batch size."""
        return tf.shape(self._inputs)[0]
    
    @property
    def reg_lambda(self):
        """Gets the regularization factor (lambda) for weight decay."""
        return self._reg_lambda