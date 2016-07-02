import os
import fnmatch


def get_filenames(root_dir, pattern):
    """Gets a list of files of a given directory matching a
    specified pattern, by using a resursive search.
    Parameters
    ----------
    root_dir: str
        The directory to recursively look into.
    pattern: str
        The file pattern search string, such as '*.jpg'.
    Returns:
    ----------
    matches: list(string)
        Returns a list of filenames that match this pattern.
    """
    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches