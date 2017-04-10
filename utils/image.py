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

    if image is None:
        raise IOError("Image file not found")
    
    # introduce a 1-channel dimension to handle the indexing
    # of color and gray images the same way
    if color_flags == cv2.IMREAD_GRAYSCALE:
        image = np.expand_dims(image, axis=2)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write(filepath, image):
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
    
    if image.shape[2] == 3:
        image = cast(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    factor = 1
    if is_float_image(image):
        factor = 255
        
    cv2.imwrite(filepath, image * factor)


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
    assert scale is None or size is None, "Either define scale or size, not both."
    assert scale is not None or size is not None, "Either scale or size parameter has to defined."
    
    img_shape = np.shape(image)
    
    if scale is not None:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    elif size is not None:
        image = cv2.resize(image, (size[1], size[0]))

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
        image = cast(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.expand_dims(image, axis=2)
    # introduce a 1-channel dimension to handle the indexing
    # of color and gray images the same way
    return image


def to_rgb(image):
    """Converts a grayscaled image to a colored one.
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
            image = cast(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def cast(image):
    """Converts an image to an OpenCV compatible type.
    Parameters
    ----------
    image: ndarray of shape [h, w, c]
        The image to ensure the correct type.
    Returns
    ---------
    The converted image or the same if the type was already correct.
    """
    if is_valid_type(image):
        return image
    elif is_uint_image(image):
        return np.uint8(image)
    else:
        return np.float32(image)

    
def is_valid_type(array):
    """Gets whether the array has a valid OpenCV image type"""
    t = array.dtype
    return True if t == np.float32 or t == np.uint8 or t == np.uint16 else False

    
def is_uint_image(image):
    """Gets whether the image is an image of scale [0, 255].
       This is only determined by the type, not by the data.
    """
    t = image.dtype
    return True if t == np.int or t == np.uint32 or t == np.int32 or t == np.int64 \
        or t == np.uint8 or t == np.uint16 else False


def is_float_image(image):
    """Gets whether the image is an image of scale [0.0, 1.0].
       This is only determined by the type, not by the data.
    """
    t = image.dtype
    return True if t == np.float or t == np.float32 or t == np.float64 else False


def pad_or_crop(images, desired_shape, pad_value=0,
                ensure_copy=True):
    """Pads or crops an image (or images) to the desired size. The padding and
       cropping is performed relative to the center.
    Parameters
    ----------
    image: ndarray
        The image (or images) to pad or crop of shape [..., h, w, c].
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
    h = images.shape[-3]
    w = images.shape[-2]
    c = images.shape[-1]

    desired_h = desired_shape[0]
    desired_w = desired_shape[1]
    
    do_pad = True if (desired_h > h or desired_w > w) else False
    do_crop = True if (desired_h < h or desired_w < w) else False
    
    if do_pad:
        pad_top = max(0, (desired_h - h) // 2)
        pad_bottom = max(0, desired_h - h - pad_top)
        pad_left = max(0, (desired_w - w) // 2)
        pad_right = max(0, desired_w - w - pad_left)
        
        pad_tuples = [(pad_top, pad_bottom),
                      (pad_left, pad_right),
                      (0, 0)]
        # insert zero-pad for all dimension prior the image shape (e.g. batch-size, time,...)
        for i in range(len(images.shape) - 3):
            pad_tuples.insert(0, (0,0))
        
        # np.pad always creates a copy of the original
        images = np.pad(images,
                        pad_tuples,
                        mode='constant',
                        constant_values=pad_value)
    if do_crop:
        left = max(0, (w - desired_w) // 2)
        top = max(0, (h - desired_h) // 2)
        if ensure_copy:
            # create a copy before returning the array-view
            images = images.copy()
        images = images[..., top:(top + desired_h), left:(left + desired_w), :]
        
    return images

def gaussian_blur(image, sigma=1.0):
    """Applies a gaussian filter to an image, by processing each channel seperately.
    Notes: we do not use: scipy.ndimage.filters.gaussian_filter(), because it makes
           the image gray-scaled, but with shape [..., 3]
    Parameters
    ----------
    image: ndarray
        The image to filter. Can have any number of channels, because each is
        processed separately.
    sigma: float
        The sigma level, where a larger number means more blur effect.
    Returns
    ----------
    The blurred image.
    """
    
    # apply equal gaussian and use ksize=0 to auto-compute it from sigma
    blurred_img = cv2.GaussianBlur(image,(0,0), sigma)
    return blurred_img