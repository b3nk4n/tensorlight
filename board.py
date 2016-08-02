import re
import math
import tensorflow as tf

TOWER_NAME = 'tower'


def _remove_tower_name(name):
    """Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
       session. This helps the clarity of presentation on tensorboard.
    Parameters
    ----------
    x: str
        The tensor name from which the op name is read and shortened.
    Returns
    ----------
    The op/variable name without a tower prefix
    """
    return re.sub('%s_[0-9]*/' % TOWER_NAME, '', name)
    

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
        tensor_name = _remove_tower_name(x.op.name)
        
        summary_name = tensor_name
        if scope is not None:
            summary_name = str(scope) + "/" + summary_name
        
        tf.histogram_summary(summary_name + '/activations', x)
        if show_sparsity:
            tf.scalar_summary(summary_name + '/sparsity', tf.nn.zero_fraction(x, name="sparsity"))
    

def loss_summary(losses, decay=0.9):
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
        Op for generating moving averages of losses. This could be used for
        managing the control dependencies in TensorFlow.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(decay, name="avg")
    loss_averages_op = loss_averages.apply(losses)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses:
        loss_name = _remove_tower_name(l.op.name)
        
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(loss_name +' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))

    return loss_averages_op


def variables_histogram_summary():
    """Creates a full histogram summary for every trainable variable.
    Returns
    ----------
    A list of string tensors that can ba added to a summary.
    """
    for var in tf.trainable_variables():
        yield tf.histogram_summary(var.op.name, var)

        
def gradients_histogram_summary(gradients):
    """Creates a histogramm summary for all given gradients.
    Parameters
    ----------
    gradients: list[(gradient, variable)]
        A list of (gradient, variable) pairs created by Optimizer.compute_gradients().
    Returns
    ----------
    A list of string tensors that can ba added to a summary.
    """
    for grad, var in gradients:
        if grad is not None:
            yield tf.histogram_summary(var.op.name + '/gradients', grad)
    
            
def conv_image_summary(tag, conv_out, padding=2):
    """Creates an image summary of the convolutional outputs
       for the first image in the batch .
    Parameters
    ----------
    tag: str or Tensor of type string
        A scalar Tensor of type string. Used to build the tag of the summary values.
        A placeholder could be used to feed in the tag name to generate multiple images,
        because using a fixed string causes to overwrite the previous one.
    conv_out: 4D Tensor of shape [batch_size, h, w, c] or 3D Tensor of shape [h, w, c]
            The convolutional output to write to summary. Note that only the first image
            of a patch is used.
    padding: int, optional
        The padding between each patch of the grid.
    """
    with tf.name_scope("conv_summary"):
        static_shape = conv_out.get_shape().as_list()
        iy = static_shape[-3] + padding * 2
        ix = static_shape[-2] + padding * 2
        channels = static_shape[-1]
        grid_length = int(math.ceil(math.sqrt(channels)))
        grid_y = grid_x = grid_length

        # slice off the first image
        co = tf.slice(conv_out, (0,0,0,0),(1,-1,-1,-1))
        
        # add padding to input kernel
        co = tf.pad(co,
                    tf.constant([[0,0],[padding,padding],[padding,padding],[0,0]]))
        
        # remove batch dimension
        co = tf.reshape(co, (iy, ix, channels))
        
        # add placeholder filters to be able to build a square
        placeholders_to_add = grid_y * grid_x - channels
        if (placeholders_to_add > 0):
            placeholders = tf.zeros((iy, ix, placeholders_to_add))
            co = tf.concat(2, [co, placeholders])

        co = tf.reshape(co, (iy, ix, grid_y, grid_x))
        co = tf.transpose(co, (2,0,3,1))
        co = tf.reshape(co, (1, grid_y * iy, grid_x * ix, 1))
        
        # scale to [0, 1]
        x_min = tf.reduce_min(co)
        x_max = tf.reduce_max(co)
        grid = (co - x_min) / (x_max - x_min)

        # write single image to summary
        tf.image_summary(tag, grid, max_images=1)


def conv_filter_image_summary(tag, kernel, padding=1):
    """Creates an image summary of the convolutional filters of the first layer.
    Parameters
    ----------
    tag: str or Tensor of type string
        A scalar Tensor of type string. Used to build the tag of the summary values.
        A placeholder could be used to feed in the tag name to generate multiple images,
        because using a fixed string causes to overwrite the previous one.
    kernel: 4D Tensor of shape [kh, kw, channels_in, filters_out]
        The convolutional filters to write to summary. Note that this is only supported
        for the frst conv-layer, which has to have an 1 or 3 as channels_in.
    padding: int, optional
        The padding between each patch of the grid.
    Example
    ----------
        conv = tt.network.conv2d("Conv1", ...)
        # Get kernel by reusing the same variable-scope
        with tf.variable_scope("Conv1", reuse=True):
            kernel = tf.get_variable("W")
        tt.board.conv_filter_image_summary("conv1_filters", kernel);
    """
    with tf.name_scope("filter_summary"):
        # X and Y dimensions, w.r.t. padding
        static_shape = kernel.get_shape().as_list()
        ky = static_shape[0] + padding * 2
        kx = static_shape[1] + padding * 2
        channels_in = static_shape[2]
        filters_out = static_shape[3]
        grid_length = int(math.ceil(math.sqrt(filters_out)))
        grid_y = grid_x = grid_length
        
        # add padding to input kernel
        k = tf.pad(kernel,
                   tf.constant([[padding,padding],[padding,padding],[0,0],[0,0]]))
          
        # add placeholder filters to be able to build a square
        placeholders_to_add = grid_y * grid_x - filters_out
        if (placeholders_to_add > 0):
            placeholders = tf.zeros((ky, kx, channels_in, placeholders_to_add))
            k = tf.concat(3, [k, placeholders])
            
        # put filters_out to the 1st dimension
        k = tf.transpose(k, (3, 0, 1, 2))
        # organize grid on Y axis
        k = tf.reshape(k, tf.pack([grid_x, ky * grid_y, kx, channels_in]))

        # switch X and Y axes
        k = tf.transpose(k, (0, 2, 1, 3))
        # organize grid on X axis
        k = tf.reshape(k, tf.pack([1, kx * grid_x, ky * grid_y, channels_in]))

        # back to normal order (not combining with the next step for clarity)
        k = tf.transpose(k, (2, 1, 3, 0))

        # to tf.image_summary order [batch_size, height, width, channels]
        k = tf.transpose(k, (3, 0, 1, 2))

        # scale to [0, 1]
        x_min = tf.reduce_min(k)
        x_max = tf.reduce_max(k)
        grid = (k - x_min) / (x_max - x_min)
        
        # write filter image to summary
        tf.image_summary(tag, grid, max_images=1)
