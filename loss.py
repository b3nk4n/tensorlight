import tensorflow as tf
import numpy as np


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
    return tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(tf.abs(img1 - img2), (-1, -2, -3))))


def _fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    Parameters
    ----------
    size: int
        The size (widht, height) of the filter kernel
    sigma: float
        The sigma of the Guassian distribution.
    Returns
    ----------
    A gaussian filter kernel.
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1,
                              -size//2 + 1:size//2 + 1]

    # expand each dims twice:
    # [h, w] -> [h, 2, channels_in=1, channels_out=1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def ssim(img1, img2, patch_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03, cs_map=False):
    """Calculates the Structural Similarity Metric corresponding to input images
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
    cs_map: Boolean, optional
        Whether to return the constrast-structure product only,
        instead of the complete SSIM.
        Basically only used internally for performance. Do not use it from the outside.
    Returns
    ----------
    value: float32
        The structural similarity metric value between both images, where '1' means
        they are identical and '0' means they are completely different.
    """
    with tf.name_scope('SSIM'):
        window = _fspecial_gauss(patch_size, sigma)
        C1 = (K1*L)**2
        C2 = (K2*L)**2
        mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
        mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2

        l_p = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        cs_p = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

        if cs_map:
            return tf.reduce_mean(cs_p)

        ssim_value =  tf.reduce_mean(l_p * cs_p)
        # enuse scale [0, 1] even with numerical instabilities
        ssim_value = tf.maximum(0.0, ssim_value)
        ssim_value = tf.minimum(1.0, ssim_value)
    return ssim_value


def ms_ssim(img1, img2, patch_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03,
            level_weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]):
    """Calculates the Multi-Scale Structural Similarity (MS-SSIM) Image
       Quality Assessment according to Z. Wang.
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
    levels = len(level_weights)
    assert levels >= 2 and levels <= 5, "Levels must be in range [2, 5]."
    
    with tf.name_scope('MS-SSIM'):
        weight = tf.constant(level_weights, dtype=tf.float32)
        mssim = None
        mcs = []
        for l in xrange(levels):
            if l == levels - 1:
                mssim = ssim(img1, img2, patch_size, sigma, L, K1, K2, cs_map=False)
            else:
                cs_map = ssim(img1, img2, patch_size, sigma, cs_map=True)
                mcs.append(cs_map)

            # ndimage.filters.convolve(img, downsample_filter, mode='reflect')
            img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
            img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

        # list to tensor of dim D+1
        mcs = tf.pack(mcs, axis=0)

        msssim_value = (tf.reduce_prod(mcs**weight[0:levels-1]) * (mssim**weight[levels-1]))
        # enuse scale [0, 1] even with numerical instabilities
        msssim_value = tf.maximum(0.0, msssim_value)
        msssim_value = tf.minimum(1.0, msssim_value)
    return msssim_value


def ss_ssim(img1, img2, patch_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03, level=2):
    """Calculates the Single-Scale Structural Similarity (SS-SSIM) Image
       Quality Assessment according to Z. Wang.
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
    with tf.name_scope('SS-SSIM'):
        # down sampling
        for l in xrange(level - 1):
            # ndimage.filters.convolve(img, downsample_filter, mode='reflect')
            img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
            img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

        ssssim_value = ssim(img1, img2, patch_size, sigma, L, K1, K2)
    return ssssim_value