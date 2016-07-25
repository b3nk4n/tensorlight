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
