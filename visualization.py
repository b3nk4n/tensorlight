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

# Helper functions for TF Graph visualization

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def
  

def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:100%;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def montage_batch(images):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders by Parag K. Mital, Jan 2016
    Parameters
    ----------
    batch : Tensor
        Input tensor to create montage of.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones(
        (images.shape[1] * n_plots + n_plots + 1,
         images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter, ...]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w, :] = this_img
    return m


# %%
def montage(W):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders by Parag K. Mital, Jan 2016
    Parameters
    ----------
    W : Tensor
        Input tensor to create montage of.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m


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
    img_data = np.uint8(img_array)
    f = StringIO()
    img_data = utils.image.to_rgb(img_data)
    arr = PIL.Image.fromarray(img_data)
    arr.save(f, format)
    return Image(data=f.getvalue())


def display_image(image):
    """Display an image object.
    Parameters
    ----------
    image : IPython.display.Image
        The image to display.
    """
    display(image)


def display_batch(img_array_batch, nrows=2, ncols=2):
    """Display a batch of images given as a 4D numpy array.
    Parameters
    ----------
    img_array_batch : numpy.ndarray
        The image numpy data in format [batch_size, height, width, channels]
        or a list of numpy arrays in format [height, width, channels],
        which can have 1 or 3 color channels.
        The data values have to be in range [0,1].
    nrows : uint, optional
        The number or rows.
     ncols : uint, optional
        The number or colums.
    """
    # create figure with random id
    fig = plt.figure(random.randint(1, sys.maxint))
    
    for i in xrange(min(nrows * ncols, len(img_array_batch))):
        current_img = img_array_batch[i]
        
        if len(current_img.shape) > 2 and current_img.shape[2] == 3:
            cmap = plt.cm.rgb
        else:
            if len(current_img.shape) > 2:
                current_img=np.squeeze(current_img)
            cmap = plt.cm.gray
        
        plt.subplot(nrows,ncols,i + 1)
        plt.imshow(current_img, cmap=cmap)


def display_array(img_array, format='png'):
    """Display an image object from a given numpy array.
    Parameters
    ----------
    img_array : numpy.ndarray
        The image data, which can have 1 or 3 color channels.
        The data values have to be in range [0,255].
    format : str, optional
        The image format.
    """
    image = image_from_array(img_array, format)
    display(image)