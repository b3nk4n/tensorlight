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
    """Rooted mean squared error (MSE) between images.
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


def ssim(img1, img2, patch_size=11, sigma=1.5, cs_map=False):
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
    cs_map: Boolean, optional
        Whether to return the cs mapping as well. Basically only used internally.
    Returns
    ----------
    value: float32
        The structural similarity metric value between both images, where '1' means
        they are identical and '0' means they are completely different.
    """
    window = _fspecial_gauss(patch_size, sigma)
    K1 = 0.01
    K2 = 0.03
    #L = 255 #bitdepth of image
    L = 1
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
    
    ssim_value = tf.reduce_mean(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                                (sigma1_sq + sigma2_sq + C2)))
    if cs_map:
        cs_map_value = tf.reduce_mean((2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
        return ssim_value, cs_map_value
    else:
        return ssim_value


def ms_ssim(img1, img2, patch_size=11, sigma=1.5, level=5):
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
    level: int, optional
        The number of scale levels. Must be in range [2, 5].
        We do not allow level=1, because then ssid() should be used for efficiency.
        We do not allow level>5, because empirical weights higher levels are missing.
    Returns
    ----------
    value: float32
        The multi-scale structural similarity metric value between both images,
        where '1' means they are identical and '0' means they are completely different.
    """
    assert level >= 2 and level <= 5, "Level must be in range [2, 5]."
    
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in xrange(level):
        ssim_map, cs_map = ssim(img1, img2, patch_size, sigma, cs_map=True)
        mssim.append(ssim_map)
        mcs.append(cs_map)
        img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    return tf.reduce_mean(value)