import torch

def assign_to_device(cuda_dev_id):
    """
    returns the device to use for computation, cuda if possible and the allocated request is correct, cpu otherwise
    :param cuda_dev_id: id of the desired cusa device, it could be a string or an int
    :return: the device to use for computation
    """
    if torch.cuda.is_available() is False:
        return torch.device("cpu")
    elif cuda_dev_id is None or int(cuda_dev_id) >= torch.cuda.device_count():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            return torch.device("cuda:" + str(cuda_dev_id) if torch.cuda.is_available() else "cpu")
        except ValueError:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")