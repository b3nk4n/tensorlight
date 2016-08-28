import os


def set_cuda_devices(device_list):
    """Masks the CUDA visible devices.
       Warning: Running this while another script is executed
                might end up in the other script to crash.
    Parameters
    ----------
    mask_string: list(int)
        A list of of CUDA devices, such as [0,1].
    """
    mask = ','.join(str(d) for d in device_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = mask
    

def get_cuda_devices():
    """Gets the CUDA visible devices mask.
    Returns
    ----------
    The CUDA devices as list of .
    """
    mask = os.environ['CUDA_VISIBLE_DEVICES']
    device_list = [int(d.strip()) for d in mask.split(',')]
    return device_list