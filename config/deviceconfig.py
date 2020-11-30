import torch

def get_torch_version_tuple():
    return tuple((int(x) for x in torch.__version__.split(".")[:2]))