import cv2
import math
import numpy as np
from scipy.misc import imresize


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
            image = imresize(image, scale)
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
    def __init__(self, filename_no_ext,
                 fps=24.0, frame_size=(240, 320), is_color=True):
        """Creates a VideoWriter instance.
        Parameters
        ----------
        filename_no_ext: str
            The file path to store the video, without file extension.
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
            '{}.avi'.format(filename_no_ext),
            fourcc, fps, 
            (max(VideoWriter.MIN_WIDTH, frame_size[1]),
             max(VideoWriter.MIN_HEIGHT, frame_size[0])))
        
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