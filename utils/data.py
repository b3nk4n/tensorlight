import os
import sys
import rarfile
import tarfile
import zipfile
from six.moves import urllib


EXT_RAR = ".rar"
EXT_TAR_GZ = ".tar.gz"
EXT_ZIP = ".zip"


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
                tar.extractall(target_dir)
        elif filename.endswith(EXT_ZIP):
            with zipfile.ZipFile(filepath, 'r') as zfile:
                zfile.extractall(target_dir)
        else:
            raise ValueError('File type not supported.')
        print('Successfully extracted file {}.'.format(filename))
    else:
        print('File {} has already been extracted.'.format(filename))
    
    return unzipped_dirpath