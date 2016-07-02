import os
import sys
import rarfile
import tarfile
from six.moves import urllib


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


def extract(filepath, target_dir):
    """Extracts the fiven file to the specified directory.
    Parameters
    ----------
    filepath: str
        The path to the file to extract.
    target_dir: str, optional
        The target directory to store the extracted file, relative
        to the current working directory.
    Returns
    ----------
    extracted_dir: str
        The path of the extracted file.
    """
    filename = os.path.basename(filepath) 
    filepath_no_ext = os.path.splitext(filepath)[0]   
    if not os.path.exists(filepath_no_ext):
        print('Extracting...')
        if filename.endswith('.rar'):
            rar = rarfile.RarFile(filepath)
            rar.extractall(target_dir)
            extracted_name = os.path.splitext(filename)[0]
        elif filename.endswith('.tar.gz'):
            tar = tarfile.open(filepath, 'r:gz')
            tar.extractall(target_dir)
            extracted_name = os.path.splitext(os.path.splitext(filename)[0])[0]
        else:
            raise ValueException('File type not supported.')
        print('Successfully extracted file {}.'.format(filename))
    else:
        print('File {} has already been extracted.'.format(filename))
    extracted_dir = os.path.join(target_dir, extracted_name)
    return extracted_dir