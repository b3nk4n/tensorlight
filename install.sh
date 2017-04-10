#!/usr/bin/bash

sudo apt-get install ffmpeg

sudo pip install rarfile
sudo pip install opencv-python
sudo pip install moviepy
sudo pip install h5py
sudo pip install jsonpickle
sudo pip install scipy
sudo pip install scikit-image

# TODO add mising video files in video-utils (any freeware videos)
# TODO test the video-utils script once more :)
# TODO make the progress indicator runnable on python-terminal (or detect the environment and run an alternative in this case?)
# TODO make model-def script and dataset-def script compatible to tensorflor-learn (if possible)
# TODO create a setup.py script or another standard install script to install al dependencies in a single call
