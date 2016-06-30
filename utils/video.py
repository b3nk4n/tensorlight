import cv2


class VideoReader():
    """Video file reader class using OpenCV."""
    def __init__(self, videofile, from_time=0):
        self.vidcap = cv2.VideoCapture(videofile)
        if from_time != 0:
            self.vidcap.set(cv2.CAP_PROP_POS_MSEC, from_time)
        
    def __enter__(self):
        """Enters the context manager."""
        return self
    
    def __exit__(self, type, value, traceback):
        """Exits the context manager and releases the video."""
        self.release()
        
    def next_frame(self, scale=1.0):
        """Reads the next from from the video.
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
        
    def release(self):
        """Releases the video file resources."""
        self.vidcap.release()
