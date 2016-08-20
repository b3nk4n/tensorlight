import os
import cv2
import numpy as np


VALUE_RANGE_0_1 = '0,1'
VALUE_RANGE_1 = '-1,1'
VALUE_RANGE_0_255 = '0,255'


def read(filepath, color_flags=cv2.IMREAD_COLOR):
    """Opens an image file from disk.
    Parameters
    ----------
    filepath: str
        The path to the image file to open.
    color_flags: int, optional
        Flag that indicates the color of the image. Most common
        values are cv2.IMREAD_GRAYSCALE or cv2.IMREAD_COLOR.
    Returns
    ----------
    image: ndarray(uint8)
        Returns image data as an array of shape [height, width, channels].
    """
    image = cv2.imread(filepath, flags=color_flags)
    # introduce a 1-channel dimension to handle the indexing
    # of color and gray images the same way
    if color_flags == cv2.IMREAD_GRAYSCALE:
        image = np.expand_dims(image, axis=2)
    return image


def write(filepath, image, value_range=VALUE_RANGE_0_255):
    """Saves an image or a frame to the specified path.
    Parameters
    ----------
    filepath: str
        The path to the file.
    image: ndarray(float/int)
        The image data.
    value_range: int (e.g. VALUE_RANGE_0_1)
        The value range of the provided image data.
    """
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    if value_range == VALUE_RANGE_0_1:
        image = image * 255.0
    elif value_range == VALUE_RANGE_1:
        image = image * 127.5 + 127.5
    elif value_range == VALUE_RANGE_0_255:
        pass  
    cv2.imwrite(filepath, image)


def write_as_binary(filepath, image):
    """Saves an image as a binary file to the specified path.
    Parameters
    ----------
    filepath: str
        The path to the file.
    image: ndarray(float/int)
        The image data.
    """
    with open(filepath, "w") as f:
        image_bytes = image.tobytes()
        f.write(image_bytes)


def resize(image, scale=None, size=None):
    """Resizes the image data by the given scale.
    Parameters
    ----------
    image: ndarray(uint8)
        The image to rescale
    scale: float, optional
        The scale factor, where 1.0 means no change.
    size: tuple(int), optional
        The size of the resized image is a tuple of 2 values (height, width).
    Returns
    ----------
    image: ndarray(uint8)
        Returns the rescaled image.
    """
    img_shape = np.shape(image)
    
    if scale is not None:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    elif size is not None:
        image = cv2.resize(image, (size[1], size[0]))
    else:
        raise ValueError('Either scale or size parameter has to defined.')
    
    if img_shape[2] == 1:
        image = np.expand_dims(image, axis=2)
     
    return image


def to_grayscale(image):
    """Converts a colored image to a grayscaled one.
    Parameters
    ----------
    image: ndarray()
        An image with the shape of [height, width, 3].
    Returns
    ---------
    image: ndarray(uint8)
        Returns a grayscaled image with shape [height, width, 1].
    """
    img_channels = np.shape(image)[2]
    if img_channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=2)
    # introduce a 1-channel dimension to handle the indexing
    # of color and gray images the same way
    return image


def to_rgb(image):
    """Converts a grayscaled image with scale [0,1] to a colored one.
    Parameters
    ----------
    image: ndarray(uint8)
        A grayscaled image with the shape of [height, width, 1]
        or of shape [height, widht].
    Returns
    ---------
    image: ndarray(uint8)
        Returns a converted image with shape [height, width, 3].
    """
    image_shape = image.shape
    
    if len(image_shape) > 2:
        img_channels = image_shape[2]
        if img_channels == 1:
            image = np.squeeze(image, axis=2)
   
        if img_channels != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
