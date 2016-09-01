import numpy as np
import tensorflow as tf


def random_distortion(image, contrast_lower=0.2, contrast_upper=1.8, brightness_max_delta=0.2, seed=None):
    """Distorts a single image for data augmentation, by applying random horizontal flipping,
       contrast and brightness.
    Parameters
    ----------
    image: Tensor with shape [height, width, channels]
        The image to apply distortion.
    contrast_lower: float, optional
        The lower contrast level, relative to the normal contrast of 1.0.
    contrast_upper: float, optional
        The upper contrast level, relative to the normal contrast of 1.0.
    brightness_max_delta: float, optional
        Defines the max brighness delta.
    seed: int or None, optional
        Used to create a random seed. See set_random_seed for behavior.
    Returns
    ----------
    image: Tensor with same shape as input.
        The randomly distorted image.
    """
    with tf.name_scope('random_distort'):
        image = tf.image.random_flip_left_right(image,
                                                seed=seed)
        # Because these operations are not commutative, consider randomizing the order their operation:
        image = tf.image.random_contrast(image,
                                         lower=contrast_lower,
                                         upper=contrast_upper,
                                         seed=seed + 1 if seed else None)
        image = tf.image.random_brightness(image,
                                           max_delta=brightness_max_delta,
                                           seed=seed + 2 if seed else None)
    return image


def equal_random_distortion(images, contrast_lower=0.2, contrast_upper=1.8, brightness_max_delta=0.2, seed=None):
    """Distorts a list of images equally for data augmentation, by applying random horizontal flipping,
       contrast and brightness.
    Parameters
    ----------
    images: Tensor list with shape list([height, width, channels])
        The images to apply an equal distortion to.
    contrast_lower: float, optional
        The lower contrast level, relative to the normal contrast of 1.0.
    contrast_upper: float, optional
        The upper contrast level, relative to the normal contrast of 1.0.
    brightness_max_delta: float, optional
        Defines the max brighness delta.
    seed: int or None, optional
        Used to create a random seed. See set_random_seed for behavior.
    Returns
    ----------
    image: Tensor with same shape as input.
        The randomly distorted image.
    Raises
    ----------
    ValueError: If `contrast_upper <= contrast_lower`, if `contrast_lower < 0`
                or if `brightness_max_delta` is negative.
    """
    if contrast_upper <= contrast_lower:
        raise ValueError('contrast_upper must be > lower.')

    if contrast_lower < 0:
        raise ValueError('contrast_lower must be non-negative.')
        
    if brightness_max_delta < 0:
        raise ValueError('brightness_max_delta must be non-negative.')
    
    with tf.name_scope('eq_random_distort'):
        uniform_random = tf.random_uniform([], 0, 1.0, dtype=tf.float32, seed=seed)
        mirror = tf.less(tf.pack([1.0, uniform_random, 1.0]), 0.5)

        contrast_factor = tf.random_uniform([], contrast_lower, contrast_upper,
                                            seed=seed + 1 if seed else None)
        delta = tf.random_uniform([], -brightness_max_delta, brightness_max_delta,
                                  seed=seed + 2 if seed else None)

        for i in xrange(len(images)):
            images[i] = tf.reverse(images[i], mirror)
            images[i] = tf.image.adjust_contrast(images[i], contrast_factor)
            images[i] = tf.image.adjust_brightness(images[i], delta)
    return images


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
    with tf.name_scope('fspecial_gauss'):
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
        gauss = g / tf.reduce_sum(g)
    return gauss


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
       Problem:
           In TensorFlow 0.10: reduce_prod() causes Error while back-prop.
           Wrap the Optimizer with tf.device('/cpu:0')!
       Solution:
           https://github.com/girving/tensorflow/commit/957fe73546f25c7f88232ed560ed285c1c97067d
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

    with tf.name_scope('MSSSIM'):
        weights = tf.constant(level_weights, dtype=tf.float32, name="level_weights")
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

        # Note: In TensorFlow 0.10: reduce_prod() causes Error while back-prop.
        #       Wrap the Optimizer with tf.device('/cpu:0')!
        msssim_val = tf.reduce_prod((mcs**weights[0:levels-1]) * (mssim**weights[levels-1]), name='prod')
        # enuse scale [0, 1] even with numerical instabilities
        msssim_val = tf.maximum(0.0, msssim_val)
        msssim_val = tf.minimum(1.0, msssim_val)
        msssim_value = msssim_val
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
    with tf.name_scope('SSSSIM'):
        # down sampling
        for l in xrange(level - 1):
            # ndimage.filters.convolve(img, downsample_filter, mode='reflect')
            img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
            img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

        ssssim_value = ssim(img1, img2, patch_size, sigma, L, K1, K2)
    return ssssim_value