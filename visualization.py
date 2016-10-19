# NOTE:
# To force matplotlib to use inline rendering, insert
# the following line inside the ipython notebook:
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import os
import sys
import random
from cStringIO import StringIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import tensorflow as tf
import utils


def image_from_array(img_array, format='png'):
    """Creates an image object from a given numpy array.
    Parameters
    ----------
    img_array : numpy.ndarray
        The image data, which can have 1 or 3 color channels.
    Returns
    -------
    IPython.display.Image
        An image object for plots.
    """
    factor = 1
    if utils.image.is_float_image(img_array):
        factor = 255
    
    img_data = np.uint8(img_array * factor)
    f = StringIO()
    img_data = utils.image.to_rgb(img_data)
    arr = PIL.Image.fromarray(img_data)
    arr.save(f, format)
    return Image(data=f.getvalue())


def display_image(image):
    """Display an image object.
    Remarks: Some RGB images might be displayed with changed colors.
    Parameters
    ----------
    image : IPython.display.Image
        The image to display.
    """
    if image is None:
        return
    
    display(image)


def display_batch(img_array_batch, nrows=2, ncols=2, title=''):
    """Display a batch of images given as a 4D numpy array.
    Remarks: Some RGB images might be displayed with changed colors.
    Parameters
    ----------
    img_array_batch : numpy.ndarray
        The image numpy data in format [batch_size, height, width, channels]
        or a list of numpy arrays in format [height, width, channels],
        which can have 1 or 3 color channels.
    nrows : uint, optional
        The number or rows.
    ncols : uint, optional
        The number or colums.
    title: str, optional
        The title of the figure.
    """
    if img_array_batch is None:
        return
    
    # determine scale from fist image
    if (utils.image.is_float_image(img_array_batch[0])):
        max_value = 1
    else:
        max_value = 255
    
    # create figure with random id
    fig = plt.figure(random.randint(1, sys.maxint))
    fig.suptitle(title, fontsize=12, fontweight='semibold')
    for i in xrange(min(nrows * ncols, len(img_array_batch))):
        current_img = img_array_batch[i]
        
        if len(current_img.shape) > 2 and current_img.shape[2] == 3:
            cmap = None
        else:
            if len(current_img.shape) > 2:
                current_img=np.squeeze(current_img)
            cmap = plt.cm.gray
        
        ax = plt.subplot(nrows,ncols,i + 1)
        plt.imshow(current_img, cmap=cmap, vmin=0, vmax=max_value)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def display_array(img_array, format='png'):
    """Display an image object from a given numpy array.
    Remarks: Some RGB images might be displayed with changed colors.
    Parameters
    ----------
    img_array : numpy.ndarray
        The image data, which can have 1 or 3 color channels.
        The data values have to be in range [0,255].
    format : str, optional
        The image format.
    """
    if img_array is None:
        return
    
    image = image_from_array(img_array, format)
    display(image)
