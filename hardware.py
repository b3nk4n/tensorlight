import os


def setCudaDeviceMask(mask_string):
    """Masks the CUDA visible devices.
       Warning: Running this while another script is executed
                might end up in the other script to crash.
    Parameters
    ----------
    mask_string: str
        A comma seperated string of CUDA devices, such as '0,1'.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = mask_string
    

def getCudaDeviceMask():
    """Gets the CUDA visible devices mask.
    Returns
    ----------
    The CUDA device mask as a string.
    """
    os.environ['CUDA_VISIBLE_DEVICES']