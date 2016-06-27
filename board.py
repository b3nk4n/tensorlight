import re
import tensorflow as tf

TOWER_NAME = 'tower'


def activation_summary(x):
    """Creates a summary for an activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Parameters
    ----------
    x: Tensor
        The tensor to write the activation summary for.
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    

def loss_summary(losses, decay=0.99):
    """Add summaries for losses in the used model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Parameters
    ----------
    losses: list(Tensor)
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
