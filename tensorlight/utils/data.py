import os
import sys
import rarfile
import tarfile
import zipfile
from six.moves import urllib

import numpy as np
import tensorlight as tt


EXT_RAR = ".rar"
EXT_TAR_GZ = ".tar.gz"
EXT_ZIP = ".zip"

SUBDIR_TRAIN = '_train'
SUBDIR_VALID = '_valid'
SUBDIR_TEST = '_test'


def download(url, target_dir):
    """Downloads a file from a given URL to the specified directory while indicating
    the progress in the command line.
    Parameters
    ----------
    url: str
        The URL to download the file from.
    target_dir: str
        The target directory to store the downloaded file.
    Returns
    ----------
    filepath: str
        The path of the downloaded file.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filename = os.path.basename(url)
    filepath = os.path.join(target_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(url, filepath, reporthook=_progress)
        print()

        statinfo = os.stat(filepath)
        print('Successfully downloaded file {}: {} bytes'.format(filename, statinfo.st_size))
    else:
        print('File {} has already been downloaded.'.format(filename))
    return filepath


def extract(filepath, target_dir, unpacked_name=None):
    """Extracts the fiven file to the specified directory.
    Parameters
    ----------
    filepath: str
        The path to the file to extract.
    target_dir: str, optional
        The target directory to store the extracted file, relative
        to the current working directory.
    unpacked_name: str, optional,
        Because it seams to be hard to find the name of the (previously)
        extracted folder, this gives the possibility to set the name manually
        to ensure that the file is not extracted again.
    Returns
    ----------
    extracted_dir: str
        The path of the extracted file.
    """
    filename = os.path.basename(filepath) # xxx.tar.gz
    
    if unpacked_name is None:
        unzipped_dirpath = os.path.splitext(filepath)[0]
        unzipped_dirpath = os.path.splitext(filepath)[0] # do it twice to support: .tar.gz
    else:
        unzipped_dirpath = os.path.join(target_dir, unpacked_name)

    if not os.path.exists(unzipped_dirpath):
        print('Extracting...')
        if filename.endswith(EXT_RAR):
            with rarfile.RarFile(filepath) as rar:
                rar.extractall(target_dir)
        elif filename.endswith(EXT_TAR_GZ):
            with tarfile.open(filepath, 'r:gz') as tar:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, target_dir)
        elif filename.endswith(EXT_ZIP):
            with zipfile.ZipFile(filepath, 'r') as zfile:
                zfile.extractall(target_dir)
        else:
            raise ValueError('File type not supported.')
        print('Successfully extracted file {}.'.format(filename))
    else:
        print('File {} has already been extracted.'.format(filename))
    
    return unzipped_dirpath


def as_binary(array):
    """Creates a copy of an array, where the data is binary but of the same type.
       All values >= 0.5 are considered to be 1.0, else 0.0
    Parameters
    ----------
    array: numpy array
        The numpy array in a float-range of [0.0, 1.0].
    Returns
    ----------
    The converted array as float only containing values of {0.0, 1.0}
    """
    return np.around(array)


def preprocess_videos(dataset_path, subdir, file_list, image_size, serialized_sequence_length,
                      gray_scale=False, scale_factor=1.0):
    """Serializes frame sequences from a given list of videos to the specified directories,
       or retrieves the list of existing files if these already exist.
    Parameters
    ----------
    dataset_path: str
        The root folder of the video data.
    subdir: str
        The subdir, basically to seperate training, validation and test data.
        It is recommended to use the constants, e.g tt.utils.data.SUBDIR_TRAIN.
    file_list: list(str)
        The list of all files, using the relative file path.
    image_size: int list or tuple of shape [h, w, c]
        The image shape of the video data. All videos will be cropped or padded relative
        to this value.
    serialized_sequence_length: int
        The frame length of each serialized bundle. Each video leads to multiple bundles.
        The serialized bundles are non-overlapping and an incomplete frame-bundle will be
        dismissed.
    scale_factor: float in range (0.0, 1.0]
        The scale factor of the video. Take care to use a factor that resizes the video evenly.
    Returns
    ----------
    (dataset_size, seq_file_list): as type (int, list(str)).
        The dataset size and a list with the path to the serialized sequence bundles.
    """
    assert scale_factor > 0 and scale_factor <= 1, "Scale factor has to be in range (0.0, 1.0]."
    
    target_size = [int(image_size[0] * scale_factor),
                   int(image_size[1] * scale_factor),
                   1 if gray_scale else image_size[2]]
    
    # create additional subdir for the image size that a
    # previous preprocessing does not have to be deleted
    image_prop_dir = "{}_{}_{}".format(target_size[0], target_size[1], target_size[2])
    full_path = os.path.join(dataset_path, subdir, image_prop_dir)
    seq_file_list = tt.utils.path.get_filenames(full_path, '*.seq')
    
    # Reuse previous preprocessing if possible
    files_count = len(seq_file_list)
    if files_count > 0:
        print("Found {} serialized frame sequences. Skipping serialization.".format(files_count))
        return files_count, seq_file_list          
                
    # create subdir folder that will contain the .seq files
    if not os.path.exists(full_path):
        os.makedirs(full_path)            
    
    print("Serializing frame sequences to '{}'...".format(full_path))
    success_counter = 0
    short_counter = 0
    progress = tt.utils.ui.ProgressBar(len(file_list))
    for i, filename in enumerate(file_list):
        with tt.utils.video.VideoReader(filename) as vr:
            # until we reach the end of the video
            clip_id = 0
            while True:
                frames = []
                if vr.frames_left >= serialized_sequence_length:
                    for f in xrange(serialized_sequence_length):
                        frame = vr.next_frame(scale_factor)

                        if frame is None:
                            break
                        
                        # ensure bounds
                        frame = tt.utils.image.pad_or_crop(frame, target_size, pad_value=0.0,
                                                           ensure_copy=False)
                        
                        # convert to gray if requried
                        if gray_scale == 1:
                            frame = tt.utils.image.to_grayscale(frame)
                            
                        frames.append(frame)

                if len(frames) == serialized_sequence_length:
                    filename = os.path.basename(filename)
                    filename_seq = "{}-{}.seq".format(os.path.splitext(filename)[0], clip_id)
                    filepath_seq = os.path.join(full_path, filename_seq)
                    tt.utils.image.write_as_binary(filepath_seq, np.asarray(frames))
                    success_counter += 1
                    clip_id += 1
                else:
                    if clip_id == 0:
                        # clip end reached in first run (video was not used at all!)
                        short_counter += 1
                    break
        progress.update(i+1)
    print("Successfully generated {} frame sequences. Too short: {}" \
          .format(success_counter, short_counter))
    
    seq_file_list = tt.utils.path.get_filenames(full_path, '*.seq')
    return success_counter, seq_file_list