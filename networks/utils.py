import torch
import torch.nn as nn
import os

from utils.file_io import load_model
from utils.misc import ensure_tuple_size


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.
    Example:
        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    assert labels.dim() > 0, "labels should have dim of 1 or more."

    # if `dim` is bigger, add singelton dim at the end
    if labels.ndimension() < dim + 1:
        shape = ensure_tuple_size(labels.shape, dim + 1, 1)
        labels = labels.reshape(*shape)

    sh = list(labels.shape)

    assert sh[dim] == 1, "labels should have a channel with length equals to one."
    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total params': total_num, 'Trainable params': trainable_num}


def build_network(net, checkpoint_path, snapshot, gpu_ids=None):
    epoch = 0
    metric = 0

    if snapshot is not None:
        print('==> load pretrained model <==')
        net, epoch, metric = load_model(net, os.path.join(checkpoint_path, snapshot), gpu_ids)
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if gpu_ids is None:
                net = nn.DataParallel(net).cuda()
            else:
                net = nn.DataParallel(net, device_ids=list(range(len(gpu_ids)))).cuda()
        elif torch.cuda.is_available():
            net = net.cuda()

    print(get_parameter_number(net))
    return net, metric, epoch
