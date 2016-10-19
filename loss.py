import numpy as np
import tensorflow as tf
import tensortools as tt


def sse(outputs, targets, name=None):
    """Sum of squared error (SSE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('SSE_loss'):
        outputs_rank = outputs.get_shape().ndims
        sum_indices = tuple(range(1, outputs_rank))
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(outputs - targets), sum_indices), name=name)

    
def mse(outputs, targets, name=None):
    """Mean squared error (MSE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('MSE_loss'):
        return tf.reduce_mean(tf.square(outputs - targets), name=name)


def rsse(outputs, targets, name=None):
    """Rooted sum of squared error (RSSE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('RSSE_loss'):
        outputs_rank = outputs.get_shape().ndims
        sum_indices = tuple(range(1, outputs_rank))
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(tf.square(outputs - targets), sum_indices)), name=name)

    
def rmse(outputs, targets, name=None):
    """Rooted mean squared error (RMSE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('RMSE_loss'):
        outputs_rank = outputs.get_shape().ndims
        reduction_indices = tuple(range(1, outputs_rank))
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_mean(tf.square(outputs - targets), reduction_indices)), name=name)


def sae(outputs, targets, name=None):
    """Sum of aboslute error (SAE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('SAE_loss'):
        outputs_rank = outputs.get_shape().ndims
        sum_indices = tuple(range(1, outputs_rank))
        return tf.reduce_mean(
            tf.reduce_sum(tf.abs(outputs - targets), sum_indices), name=name)
    
    
def mae(outputs, targets, name=None):
    """Mean aboslute error (MAE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('MAE_loss'):
        return tf.reduce_mean(tf.abs(outputs - targets), name=name)


def rsae(outputs, targets, name=None):
    """Rooted sum of absolute error (RSAE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('RSAE_loss'):
        outputs_rank = outputs.get_shape().ndims
        sum_indices = tuple(range(1, outputs_rank))
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(tf.abs(outputs - targets), sum_indices)), name=name)
    
    
def rmae(outputs, targets, name=None):
    """Rooted mean absolute error (RMAE) between images.
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('RMAE_loss'):
        outputs_rank = outputs.get_shape().ndims
        reduction_indices = tuple(range(1, outputs_rank))
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_mean(tf.abs(outputs - targets), reduction_indices)), name=name)
    
    
def bce(output_probs, targets, from_logits=False, name=None):
    """Binary cross-entropy (BCE) between an output and a target tensor.
       Remarks: In case of images, this loss gives great results for image
                like MNIST or MovingMNIST, but does NOT work for natural images
                with color or gray-scaled, as it can lead to negative loss.
       References:
           Taken from Keras implementation (TensorFlow backend).
    Parameters
    ----------
    output_probs: Tensor [batch_size, ...] of type float32
        The probabilities of the output. It should be the output of tf.sigmoid(output).
    img2: Tensor [batch_size, ...] of type float32
        The probabilities of the output in scale [0, 1].
    from_logits: Boolean, optional
        Whether the given values are probabilites (default) or logits.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the caluclated error.
    """
    with tf.name_scope('BCE_loss'):
        # flatten
        output_probs_flat = tf.contrib.layers.flatten(output_probs)
        targets_flat = tf.contrib.layers.flatten(targets)

        if not from_logits:
            # transform back to logits
            EPSILON = 10e-8
            output_probs_flat = tf.clip_by_value(output_probs_flat, EPSILON, 1 - EPSILON)
            output_probs_flat = tf.log(output_probs_flat / (1 - output_probs_flat))
        bce_values = tf.nn.sigmoid_cross_entropy_with_logits(output_probs_flat, targets_flat)
        return tf.reduce_mean(bce_values, name=name)

    
def ce(outputs, targets, name=None):
    """Cross entropy error (CE).
    Parameters
    ----------
    outputs: Tensor [batch_size, ...] of type float32
        The first tensor.
    targets: Tensor [batch_size, ...] of type float32
        The second tensor.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('CE_loss'):
        outputs_rank = outputs.get_shape().ndims
        sum_indices = tuple(range(1, outputs_rank))
        return -tf.reduce_mean(
            tf.reduce_sum(targets * tf.log(outputs), sum_indices), name=name)
    

def ssim(img1, img2, patch_size=11, sigma=1.5, L=1.0, K1=0.01, K2=0.03, name=None):
    """Calculates the Structural Similarity loss
       Reference: 
           This function attempts to mimic precisely the functionality of ssim.m a
           MATLAB provided by the author's of SSIM
           https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image. Expected to have 1 channel and in scale [0, 1].
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image. Expected to have 1 channel and in scale [0, 1].
    patch_size: int, optional
        The size of a single patch.
    sigma: float, optional
        The Gaussian's sigma value.
    L: int, optional
        Note: Not using 255 will result in slightly different result.
        The bit depth of the image. Use '1' when a value scale of [0,1] is used.
        The scale of [-1, 1] is not supported and has to be rescaled.
    K1: float, optional
        The K1 value.
    K2: float, optional
        The K2 value.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    value: float32
        The structural similarity loss value between both images.
    """
    with tf.name_scope('SSIM_loss'):
        return 1 - tt.image.ssim(img1, img2, patch_size, sigma,
                                 L, K1, K2, name=name)


def ms_ssim(img1, img2, patch_size=11, sigma=1.5, L=1.0, K1=0.01, K2=0.03,
            level_weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], name=None):
    """Calculates the Multi-Scale Structural Similarity (MS-SSIM) loss.
       References:
            Z. Wang's "Multi-scale structural similarity
            for image quality assessment" Invited Paper, IEEE Asilomar Conference on
            Signals, Systems and Computers, Nov. 2003

            Author's MATLAB implementation:-
            http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image. Expected to have 1 channel and in scale [0, 1].
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image. Expected to have 1 channel and in scale [0, 1].
    patch_size: int, optional
        The size of a single patch.
    sigma: float, optional
        The Gaussian's sigma value.
    L: int, optional
        WARNING: NOT USING 255 WILL RESULT IN DIFFERENT RESULTS!
        The bit depth of the image. Use '1' when a value scale of [0,1] is used.
        The scale of [-1, 1] is not supported and has to be rescaled.
    K1: float, optional
        The K1 value.
    K2: float, optional
        The K2 value.
    level_weights: list(float), optional
        The weights for each scale level M. Must be in range [2, 5].
        We do not allow level=1, because then ssid() should be used for efficiency.
        We do not allow level>5, because empirical weights higher levels are missing.
        If a different value is selected, other weights should be used, because the
        default values have been obtained from an empirical analysis. A level of 5 is only
        suitable for huge images. E.g an image of 64x64 pixels with level M=3 can result
        in NaN values.
        It can be considered to allow more levels with smaller patch_size (5,7,9). Some other
        papers use smaller sizes. Also, when in the non-human-perception optimized setting, all
        wheits are equal with SUM(level_weights)=1.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    value: float32
        The multi-scale structural similarity metric value between both images,
        where '1' means they are identical and '0' means they are completely different.
    """
    with tf.name_scope('MSSSIM_loss'):
        return 1 - tt.image.ms_ssim(img1, img2, patch_size, sigma,
                                    L, K1, K2, level_weights, name=name)


def ss_ssim(img1, img2, patch_size=11, sigma=1.5, L=1.0, K1=0.01, K2=0.03, level=2, name=None):
    """Calculates the Single-Scale Structural Similarity (SS-SSIM) loss.
       References:
            Z. Wang's "Multi-scale structural similarity
            for image quality assessment" Invited Paper, IEEE Asilomar Conference on
            Signals, Systems and Computers, Nov. 2003
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image. Expected to have 1 channel and in scale [0, 1].
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image. Expected to have 1 channel and in scale [0, 1].
    patch_size: int, optional
        The size of a single patch.
    sigma: float, optional
        The Gaussian's sigma value.
    L: int, optional
        WARNING: NOT USING 255 MIGHT RESULT IN DIFFERENT RESULTS!
        The bit depth of the image. Use '1' when a value scale of [0,1] is used.
        The scale of [-1, 1] is not supported and has to be rescaled.
    K1: float, optional
        The K1 value.
    K2: float, optional
        The K2 value.
    level: int, optional
        The level M=2.
        A level of M=1 equals simple ssim() function.
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    value: float32
        The single-scale structural similarity metric value between both images,
        where '1' means they are identical and '0' means they are completely different.
    """
    with tf.name_scope('SSSSIM_loss'):
        return 1 - tt.image.ss_ssim(img1, img2, patch_size, sigma,
                                    L, K1, K2, level, name=name)

    
def _gradient_differences(img1, img2):
    """Computs the gradient differences between two images.
       Based on: https://arxiv.org/abs/1511.05440 which is optimized and simplified
       for efficiency.        
    """
    shape = img1.get_shape().as_list()
        
    # gradient difference
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    pos = tf.constant(np.identity(shape[3]), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.pack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.pack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]

    img1_dx = tf.abs(tf.nn.conv2d(img1, filter_x, [1, 1, 1, 1], padding='SAME'))
    img1_dy = tf.abs(tf.nn.conv2d(img1, filter_y, [1, 1, 1, 1], padding='SAME'))
    img2_dx = tf.abs(tf.nn.conv2d(img2, filter_x, [1, 1, 1, 1], padding='SAME'))
    img2_dy = tf.abs(tf.nn.conv2d(img2, filter_y, [1, 1, 1, 1], padding='SAME'))

    grad_diff_x = tf.abs(img2_dx - img1_dx)
    grad_diff_y = tf.abs(img2_dy - img1_dy)
    return grad_diff_x, grad_diff_y 

    
def gdl(img1, img2, alpha=1.0, name=None):
    """Computes the (summed) Gradient Differences Loss (GDL) between two images on
       the same scale, as defined in: https://arxiv.org/abs/1511.05440
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image. Expected to have values in scale [0, max_value].
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image. Expected to have values in scale [0, max_value].
    alpha: float, optional
        Value that is in range [1, ...).
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    mean(sum(gdl_values)): float32 Tensor
        The per image summed Gradient Differences error over each frame in the batch.
        Attention: The value can get very large for non-similar images (>100k)
    """
    with tf.name_scope('GDL_loss'):
        grad_diff_x, grad_diff_y = _gradient_differences(img1, img2)
        gdl_values = tf.reduce_sum(grad_diff_x ** alpha + grad_diff_y ** alpha, [1, 2, 3])
        return tf.reduce_mean(gdl_values, name=name)

    
def mgdl(img1, img2, alpha=1.0, name=None):
    """Computes the Mean / per-pixel Gradient Differences Loss (GDL) between
       two images on the same scale. This version takes the mean, that values
       do not explode on large images and have a similar scale like other loss
       functions.
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image. Expected to have values in scale [0, max_value].
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image. Expected to have values in scale [0, max_value].
    alpha: float, optional
        Value that is in range [1, ...).
    name: str or None, optional
        Optioanl name to be applied in TensorBoard. Defaults to "Mean" and follows to
        '<loss-name>/Mean' in TensorBoard.
    Returns
    ----------
    mean(gdl_values): float32 Tensor
        The mean Gradient Differences error over each frame in the batch.
        Attention: The value can get very large for non-similar images (>100k)
    """
    with tf.name_scope('mGDL_loss'):
        grad_diff_x, grad_diff_y = _gradient_differences(img1, img2)
        gdl_value = tf.reduce_mean(grad_diff_x ** alpha + grad_diff_y ** alpha, name=name)
        return gdl_value