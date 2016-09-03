import os
import cv2
import math
import numpy as np
import tensortools as tt

import numpy as np
import moviepy.editor as mpy


class VideoReader():
    """Video file reader class using OpenCV."""
    def __init__(self, filename, from_time=0):
        """Creates a VideoReader instance.
        Parameters
        ----------
        filename: str
            The file path to the video.
        from_time: int, optional
            The time where to start the video from in milliseconds.
        """
        self.vidcap = cv2.VideoCapture(filename)
        if from_time != 0:
            self.vidcap.set(cv2.CAP_PROP_POS_MSEC, from_time)
        
    def __enter__(self):
        """Enters the context manager."""
        return self
    
    def __exit__(self, type, value, traceback):
        """Exits the context manager and releases the video."""
        self.release()
        
    def next_frame(self, scale=1.0):
        """Reads the next frame from the video.
        Parameters
        ----------
        scale: float, optional
            The scale value to resize the frame image.
        Returns
        ----------
        image: ndarray(uint8)
            Returns an ndarray of the image or None in case of an error.
        """
        success, image = self.vidcap.read()
        if success:
            image = tt.utils.image.resize(image, scale)
            return image
        else:
            return None
        
    def skip_frames(self, count=1):
        """Skips the next frames from the video.
        Parameters
        ----------
        count: int, optional
            The number of frames to skip.
        """
        for i in xrange(count):
            success, _ = self.vidcap.read()
            if not success:
                break
        
    def release(self):
        """Releases the video file resources."""
        self.vidcap.release()


class VideoWriter():
    MIN_WIDTH = 128
    MIN_HEIGHT = 128
    FF_MIN_BUFFER_SIZE = 16384  # from OpenCV C++ code
    
    """Video writer class using OpenCV."""
    def __init__(self, filepath,
                 fps=24.0, frame_size=(240, 320), is_color=True):
        """Creates a VideoWriter instance.
        Parameters
        ----------
        filepath: str
            The file path to store the video to write. Currently only
            the file extension ".avi" is supported.
        fps: float, optional
            The frame rate of the video in frames/seconds.
        frame_size: tuple(height,width), optional
            The frame size of the video.
        is_color: Boolean, optional
            Indicates whether the video has colors or is just gray scaled.
        """
        # Define the codec
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        self.vidwriter = cv2.VideoWriter(
            filepath,
            fourcc, fps, 
            (max(VideoWriter.MIN_WIDTH, frame_size[1]),
             max(VideoWriter.MIN_HEIGHT, frame_size[0])),
            is_color)
        
    def __enter__(self):
        """Enters the context manager."""
        return self
    
    def __exit__(self, type, value, traceback):
        """Exits the context manager and releases the video."""
        self.release()
           
    def _ensure_min_frame_size(self, frame):
        """Esures the minimum video frame size required by OpenCV.
        Parameters
        ----------
        filename: str
            The file path to store the video.
        Returns
        ----------
        frame: ndarray(uint8)
            Returns the padded video frame.
        """
        h, w, c = np.shape(frame)
        size = h * w * c
        if (size < VideoWriter.FF_MIN_BUFFER_SIZE):
            pad_top = (VideoWriter.MIN_HEIGHT - h) // 2
            pad_bottom = VideoWriter.MIN_HEIGHT - h - pad_top
            pad_left = (VideoWriter.MIN_WIDTH - w) // 2
            pad_right = VideoWriter.MIN_WIDTH - w - pad_left
            frame = np.pad(frame,
                           ((pad_top, pad_bottom),
                            (pad_left, pad_right),
                            (0, 0)),
                           mode='constant')
        return frame
        
    def write_frame(self, frame):
        """Writes a video frame to the file.
        Parameters
        ----------
        frame: ndarray(uint8)
            The video frame to write.
        """
        padded_frame = self._ensure_min_frame_size(frame)
        self.vidwriter.write(padded_frame)
  
    def release(self):
        """Releases the video file resources."""
        self.vidwriter.release()



def write_gif(filepath, images, fps=24):
    """Saves a sequence of images as an animated GIF.
    Parameters
    ----------
    filepath: str
        The filepath ending with *.gif where to save the file.
    images: list(3-D array) or 4-D array
        A list of images or a 4-D array where the first dimension
        represents the time axis.
    fps: int, optional
        The frame rate.
    """
    # to list
    if not isinstance(images, list):
        splitted = np.split(images, images.shape[0])
        images = [np.squeeze(s, axis=(0,)) for s in splitted]
      
    # ensure directory exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    clip = mpy.ImageSequenceClip(images, fps=fps)
    clip.write_gif(filepath, verbose=False)

    
def write_multi_gif(filepath, images_list, fps=24, pad_value=255, pad_width=2):
    """Saves multiple sequences of images as a single animated GIF.
       The single clips will be padded and combined in a row.
    Parameters
    ----------
    filepath: str
        The filepath ending with *.gif where to save the file.
    images_list: list(list(3-D array)) or list(4-D array)
        A list of list(images) or a list(4-D array) where the first dimension
        represents the time axis. The internal lists have to have equal length.
    fps: int, optional
        The frame rate.
    pad_value: int, optional
        The value of the image padding in range [0, 255].
    pad_width: int, optional
        The width of the padding.
    """
    for i in xrange(len(images_list)):
        # to list of list
        if not isinstance(images_list[i], list):  
            splitted = np.split(images_list[i], images_list[i].shape[0])
            images_list[i] = [np.squeeze(s, axis=(0,)) for s in splitted]
            
    for i in xrange(1, len(images_list)):
        assert len(images_list[i-1]) == len(images_list[i]), "All images-lists have to have equal length."
    
    # pad images
    for i in xrange(len(images_list)):
        for j in xrange(len(images_list[i])):
            images_list[i][j] = np.pad(images_list[i][j],
                                       ((pad_width, pad_width), (pad_width, pad_width), (0,0)),
                                       mode="constant", constant_values=pad_value)
        
    # concatenate
    concat_list = []
    for frame_idx in xrange(len(images_list[0])):
        single_frame_of_each_seq = [row[frame_idx] for row in images_list]
        concat_list.append(np.concatenate(single_frame_of_each_seq, axis=1))
    
    write_gif(filepath, concat_list, fps)

    
def _to_single_sequence(images, pad_value, pad_width, seq_length):
    # to list
    if not isinstance(images, list):
        splitted = np.split(images, images.shape[0])
        images = [np.squeeze(s, axis=(0,)) for s in splitted]
      
    # pad images
    padded_list = []
    for i in xrange(seq_length):
        if i < len(images):
            padded_list.append(np.pad(images[i],
                                      ((pad_width, pad_width), (pad_width, pad_width), (0,0)),
                                      mode="constant", constant_values=pad_value))
        else:
            shape = images[0].shape
            padded_list.append(np.ones((shape[0] + 2*pad_width, shape[1] + 2*pad_width, shape[2])) * pad_value)
            
    # concatenate
    return np.concatenate(padded_list, axis=1)
    
def write_image_sequence(filepath, images, pad_value=255, pad_width=2):
    """Saves a sequence of images as a single image file.
    Parameters
    ----------
    filepath: str
        The filepath ending with *.gif where to save the file.
    images: list(3-D array) or 4-D array
        A list of images or a 4-D array where the first dimension
        represents the time axis.
    pad_value: int, optional
        The value of the image padding in range [0, 255].
    pad_width: int, optional
        The width of the padding.
    """
    concat_image = _to_single_sequence(images, pad_value, pad_width, len(images))
    tt.utils.image.write(filepath, concat_image)
    

def write_multi_image_sequence(filepath, images_list, pad_value=255, pad_width=2):
    """Saves multiple sequences of images as a single image file.
    Parameters
    ----------
    filepath: str
        The filepath ending with *.gif where to save the file.
    images_list: list(list(3-D array)) or list(4-D array)
        A list of list(images) or a list(4-D array) where the first dimension
        represents the time axis. The internal lists have to have equal length.
    pad_value: int, optional
        The value of the image padding in range [0, 255].
    pad_width: int, optional
        The width of the padding.
    """
    max_length = 0
    for seq in images_list:
        if not isinstance(seq, list):
            max_length = max(max_length, seq.shape[0])
        else:
            max_length = max(max_length, len(seq))
    
    seq_list = []
    for i in xrange(len(images_list)):
        seq_list.append(_to_single_sequence(images_list[i], pad_value, pad_width, max_length))
        
    # concatenate
    concat_image = np.concatenate(seq_list, axis=0)
        
    tt.utils.image.write(filepath, concat_image)