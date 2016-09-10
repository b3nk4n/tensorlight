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
    # TODO: use ndarray.tofile?
    with open(filepath, "w") as f:
        image_bytes = image.tobytes()
        f.write(image_bytes)
        
        
def read_as_binary(filepath, dtype=np.uint8):
    """Reads an image as a binary file from the specified path.
    Parameters
    ----------
    filepath: str
        The path to the file.
    datatype: type
        The type/numpy.type of the data. The result is a 1D-array
        and has to be reshaped.
    Returns
    ----------
    image: ndarray(float/int)
        The image data.
    """
    return np.fromfile(filepath, dtype)


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


def pad_or_crop(image, desired_shape, pad_value=0,
                ensure_copy=True):
    """Pads or crops an image to the desired size. The padding and
       cropping is performed relative to the center.
    Parameters
    ----------
    image: ndarray
        The image to pad or crop.
    desired_shape: list or tuple of shape [h, w] or [h, w, c]
        The desired target shape. In case a shape of dimension [h, w, c]
        is passed, the channel-dim will be ignored. Is is only accepted that
        an image-shape array does not have cropped when using this function.
    pad_value: int or float, optional
        The value of the image padding, usually in range [0, 255] or [0.0, 1.0].
    ensure_copy: Boolean, optional
        If True (default) a new copy is created for the cropped image. Modifications
        on the the padded/cropped image do not influence the original. You may deactivate
        this behaviour to improve performance.
    Returns
    ----------
    The padded or cropped image.
    """
    h, w, c = image.shape
    desired_h = desired_shape[0]
    desired_w = desired_shape[1]
    
    do_pad = True if (desired_h > h or desired_w > w) else False
    do_crop = True if (desired_h < h or desired_w < w) else False
    
    if do_pad:
        pad_top = (desired_h - h) // 2
        pad_bottom = desired_h - h - pad_top
        pad_left = (desired_w - w) // 2
        pad_right = desired_w - w - pad_left
        # np.pad always creates a copy of the original
        image = np.pad(image,
                       ((pad_top, pad_bottom),
                        (pad_left, pad_right),
                        (0, 0)),
                       mode='constant',
                       constant_values=pad_value)
    if do_crop:
        left = (w - desired_w) // 2
        top = (w - desired_h) // 2
        if ensure_copy:
            # create a copy before returning the array-view
            image = image.copy()
        image = image[top:(top + desired_h), left:(left + desired_w), :]
        
    return image