import re
import math
import tensorflow as tf

TOWER_NAME = 'tower'


def activation_summary(x, show_sparsity=False, scope=None):
    """Creates a summary for an activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Parameters
    ----------
    x: Tensor
        The tensor to write the activation summary for.
    show_sparsity: Boolean, optional
        Whether to include sparsity (fraction of zeros) in the summary.
    scope: str, optional
        The scope name of the module it belongs to. This has the benefit that
        TensorBoard diagrams can be grouped together to gain a better overview.
    """
    with tf.name_scope("activation_summary"):
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        
        summary_name = tensor_name
        if scope is not None:
            summary_name = str(scope) + "/" + summary_name
        
        tf.histogram_summary(summary_name + '/activations', x)
        if show_sparsity:
            tf.scalar_summary(summary_name + '/sparsity', tf.nn.zero_fraction(x, name="sparsity"))
    

def loss_summary(losses, decay=0.99):
    """Add summaries for losses in the used model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Parameters
    ----------
    losses: list[Tensor]
        List of losses (e.g. total_loss, loss_wo_reg, reg_terms).
    decay: float, optional
        The decay to use for the exponential moving average.
    Returns
    ---------
    loss_averages_op: Tensor
        Op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    loss_averages_op = loss_averages.apply(losses)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def variables_histogram_summary():
    """Creates a full histogram summary for every trainable variable.
    """
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

        
def gradients_histogram_summary(gradients):
    """Creates a histogramm summary for all given gradients.
    Parameters
    ----------
    gradients: list[(gradient, variable)]
        A list of (gradient, variable) pairs created by Optimizer.compute_gradients().
    """
    for grad, var in gradients:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    
            
def conv_image_summary(tag, conv_out):
    """Creates an image summary of the convolutional outputs
    for the first image in the batch .
    Parameters
    ----------
    tag: str or Tensor of type string
        A scalar Tensor of type string. Used to build the tag of the summary values.
    conv_out: 4D Tensor of shape [batch_size, h, w, c] or 3D Tensor of shape [h, w, c]
            The convolutional output to write to summary. Note that only the first image
            of a patch is used.     
    """
    with tf.name_scope("conv_image_summary"):
        static_shape = conv_out.get_shape().as_list()
        iy = static_shape[-3]
        ix = static_shape[-2]
        c = static_shape[-1]

        # slice off the first image
        co = tf.slice(conv_out, (0,0,0,0),(1,-1,-1,-1))
        # remove batch dimension
        co = tf.reshape(co, (iy, ix, c))
        
        grid_length = int(math.ceil(math.sqrt(c)))

        cy = cx = grid_length
        
        placeholders_to_add = cy * cx - c
        if (placeholders_to_add > 0):
            co = tf.concat(2, [co] + ([tf.zeros((iy, ix, 1))] * placeholders_to_add))
        print(co.get_shape())
        ix += 4
        iy += 4

        co = tf.image.resize_image_with_crop_or_pad(co, iy, ix)

        co = tf.reshape(co, (iy, ix, cy, cx))
        co = tf.transpose(co, (2,0,3,1))
        co = tf.reshape(co, (1, cy * iy, cx * ix, 1))

        tf.image_summary(tag, co)
