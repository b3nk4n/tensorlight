import numpy as np
import tensorflow as tf
import tensortools as tt


def image_mse(img1, img2):
    """Mean squared error (MSE) between images.
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image.
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('MSE_loss'):
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(img1 - img2), (-1, -2, -3)))


def image_rmse(img1, img2):
    """Rooted mean squared error (RMSE) between images.
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image.
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('RMSE_loss'):
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(tf.square(img1 - img2), (-1, -2, -3))))


def image_mae(img1, img2):
    """Mean aboslute error (MAE) between images.
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image.
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('MAE_loss'):
        return tf.reduce_mean(
            tf.reduce_sum(tf.abs(img1 - img2), (-1, -2, -3)))


def image_rmae(img1, img2):
    """Rooted mean absolute error (RMAE) between images.
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image.
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image.
    Returns
    ----------
    Returns the calculated error.
    """
    with tf.name_scope('RMAE_loss'):
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(tf.abs(img1 - img2), (-1, -2, -3))))
    
    
def bce(output_probs, targets, from_logits=False):
    """Binary cross-entropy (BCE) between an output and a target tensor.
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
        return tf.reduce_mean(bce_values)


def ssim(img1, img2, patch_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03):
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
    Returns
    ----------
    value: float32
        The structural similarity loss value between both images.
    """
    with tf.name_scope('SSIM_loss'):
        return 1 - tt.image.ssim(img1, img2, patch_size, sigma, L, K1, K2)


def ms_ssim(img1, img2, patch_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03,
            level_weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]):
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
    Returns
    ----------
    value: float32
        The multi-scale structural similarity metric value between both images,
        where '1' means they are identical and '0' means they are completely different.
    """
    with tf.name_scope('MSSSIM_loss'):
        return 1 - tt.image.ms_ssim(img1, img2, patch_size, sigma, L, K1, K2, level_weights)


def ss_ssim(img1, img2, patch_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03, level=2):
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
    Returns
    ----------
    value: float32
        The single-scale structural similarity metric value between both images,
        where '1' means they are identical and '0' means they are completely different.
    """
    with tf.name_scope('SSSSIM_loss'):
        return 1 - tt.image.ss_ssim(img1, img2, patch_size, sigma, L, K1, K2, level)