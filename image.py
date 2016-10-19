import numpy as np
import tensorflow as tf
import tensorlight as light


def random_distortion(image, contrast_lower=0.8, contrast_upper=1.2, brightness_max_delta=0.2, seed=None):
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
        
        # limit to scale [0, 1]
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
        return image


def equal_random_distortion(images, contrast_lower=0.8, contrast_upper=1.2, brightness_max_delta=0.2, seed=None):
    """Distorts a list of images equally for data augmentation, by applying random horizontal flipping,
       contrast and brightness.
    Parameters
    ----------
    images: Tensor list with shape list([height, width, channels])
        The images to apply an equal distortion to in value scale [0, 1].
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
            adjusted_image = tf.reverse(images[i], mirror)
            adjusted_image = tf.image.adjust_contrast(adjusted_image, contrast_factor)
            adjusted_image = tf.image.adjust_brightness(adjusted_image, delta)
            
            # limit to scale [0, 1]
            adjusted_image = tf.minimum(adjusted_image, 1.0)
            adjusted_image = tf.maximum(adjusted_image, 0.0)
            
            images[i] = adjusted_image
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


def ssim(img1, img2, patch_size=11, sigma=1.5, L=1.0, K1=0.01, K2=0.03,
         cs_map=False, name=None):
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
    name: str or None, optional
        Optional name to be applied in TensorBoard. Defaults to the last operations name
        of this metric, such as mean, sum or min/max.
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
        ssim_value = tf.minimum(1.0, ssim_value, name=name)
    return ssim_value


def ms_ssim(img1, img2, patch_size=11, sigma=1.5, L=1.0, K1=0.01, K2=0.03,
            level_weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], name=None):
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
    name: str or None, optional
        Optional name to be applied in TensorBoard. Defaults to the last operations name
        of this metric, such as mean, sum or min/max.
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
        msssim_val = tf.minimum(1.0, msssim_val, name=name)
        msssim_value = msssim_val
        return msssim_value


def ss_ssim(img1, img2, patch_size=11, sigma=1.5, L=1.0, K1=0.01, K2=0.03, level=2,
            name=None):
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
        Optional name to be applied in TensorBoard. Defaults to the last operations name
        of this metric, such as mean, sum or min/max.
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

        ssssim_value = ssim(img1, img2, patch_size, sigma, L, K1, K2, name=name)
    return ssssim_value


def psnr(img1, img2, max_value=1.0, name=None):
    """Computes the Peak Signal to Noise Ratio (PSNR) error between two images.
       Although a higher PSNR generally indicates that the reconstruction is of higher quality,
       in some cases it may not. One has to be extremely careful with the range of validity of this metric;
       it is only conclusively valid when it is used to compare results from the same codec (or codec type)
       and same content.
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image. Expected to have values in scale [0, max_value].
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image. Expected to have values in scale [0, max_value].
    max_value: float, optional
        The maximum possible values of image intensities. Alternatively, use 255.0 for images
        in scale [0, 255].
    name: str or None, optional
        Optional name to be applied in TensorBoard. Defaults to the last operations name
        of this metric, such as mean, sum or min/max.
    Returns
    ----------
    mean(psnr_values): float32 Tensor
        The mean Peak Signal to Noise Ratio error over each frame in the batch in range [0, 99].
        Typical values for the PSNR in lossy image and video compression are between 30 and 50 dB,
        provided the bit depth is 8 bits, where higher is better. For 16-bit data typical values for
        the PSNR are between 60 and 80 dB. Acceptable values for wireless transmission quality 
        loss are considered to be about 20 dB to 25 dB.
    """
    with tf.name_scope('PSNR'):
        shape = tf.shape(img1)

        N = tf.to_float(shape[1] * shape[2] * shape[3])
        MSE = tf.reduce_sum(tf.square(img2 - img1), [1, 2, 3])

        psnr_values = 10 * light.mathex.log10(tf.square(max_value) / ((1 / N) * MSE))
        
        # define 99 as the maximum value, as values can get until infinity, as in:
        # http://stackoverflow.com/questions/26210055/psnr-of-image-using-matlab
        psnr_values = tf.minimum(99.0, psnr_values)
        
        return tf.reduce_mean(psnr_values, name=name)
    
def sharp_diff(img1, img2, max_value=1.0, name=None):
    """Computes the Sharpness Difference (Sharp. Diff.) error between between two images.
    Parameters
    ----------
    img1: Tensor [batch_size, h, w, c] of type float32
        The first image. Expected to have values in scale [0, max_value].
    img2: Tensor [batch_size, h, w, c] of type float32
        The second image. Expected to have values in scale [0, max_value].
    max_value: float, optional
        The maximum possible values of image intensities. Alternatively, use 255.0 for images
        in scale [0, 255].
    name: str or None, optional
        Optional name to be applied in TensorBoard. Defaults to the last operations name
        of this metric, such as mean, sum or min/max.
    Returns
    ----------
    mean(sdiff_values): float32 Tensor
        The mean Sharpness Differences error over each frame in the batch.
    """
    with tf.name_scope('SHARP_DIFF'):
        shape = img1.get_shape().as_list()
        
        N = tf.to_float(shape[1] * shape[2] * shape[3])

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

        img1_grad_sum = img1_dx + img1_dy
        img2_grad_sum = img2_dx + img2_dy

        grad_diff = tf.abs(img2_grad_sum - img1_grad_sum)

        sdiff_values = 10 * light.mathex.log10(max_value / ((1 / N) * tf.reduce_sum(grad_diff, [1, 2, 3])))
        
        # define 99 as the maximum value, as values can get until infinity, as we do it for PSNR
        sdiff_values = tf.minimum(99.0, sdiff_values)
        
        return tf.reduce_mean(sdiff_values, name=name)
