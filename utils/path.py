import os
import fnmatch


def get_filenames(root_dir, pattern, include_root=True):
    """Gets a list of files of a given directory matching a
       specified pattern, by using a resursive search.
    Parameters
    ----------
    root_dir: str
        The directory to recursively look into.
    pattern: str
        The file pattern search string, such as '*.jpg'.
    include_root: Boolean, optional
        Whether to include the root path or just return the
        filenames.
    Returns
    ----------
    matches: list(string)
        Returns a list of filenames that match this pattern.
    """
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            if include_root:
                filename = os.path.join(root, filename)
            matches.append(filename)
    return matches


def get_subdirnames(root_dir):
    """Gets the immediate subdirectory names of a folder.
    Parameters
    ----------
    root_dir: str
        The directory to recursively look into.
    Returns
    ----------
    A list of strings with the folder names.
    """
    return [d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))]

def get_subdirpaths(root_dir):
    """Gets the path to all subdirectories given a folder.
    Parameters
    ----------
    root_dir: str
        The directory to recursively look into.
    Returns
    ----------
    A list of strings with the folder paths.
    """
    subdir_paths = []
    for d in os.listdir(root_dir):
        path = os.path.join(root_dir, d)
        if os.path.isdir(path):
            subdir_paths.append(path)
    return subdir_paths