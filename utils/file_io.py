import os
import cv2
import torch
import numpy as np


def save_jpg(data, name):
    if data.ndim != 3:
        print('wrong input data dim when saving as jpg, the dim should equal to 3')
        os._exit(0)

    mkdir(os.path.split(name)[0] + '/')
    if data.dtype != np.ubyte:
        data_f = data.astype(np.float32)
        maxv = np.max(data_f)
        minv = np.min(data_f)
        data = (data_f - minv) / (maxv - minv) * 255.0
        data = data.astype(np.ubyte)
    if data.shape[0] == 3:
        data = data.transpose((1, 2, 0))

    cv2.imwrite(str(name) + '.jpg', data)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def partial_weight_update(model, pretrained_state):
    model_state = model.state_dict()
    state_dict = {k: v for k, v in pretrained_state.items() if k in model_state.keys()}
    model_state.update(state_dict)
    model.load_state_dict(model_state)
    return model


def load_model(net, file_path, gpu_ids=None):
    checkpoint = torch.load(file_path)
    partial_weight_update(net, checkpoint['state_dict'])
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if gpu_ids is None:
            net = torch.nn.DataParallel(net).cuda()
        else:
            net = torch.nn.DataParallel(net, device_ids=list(range(len(gpu_ids)))).cuda()
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        net = net.cuda()

    epoch = checkpoint['epoch']
    metric = checkpoint['metric']

    return net, epoch, metric


def save_model(net, metric, epoch, save_dir, tag='best_metric_model'):
    mkdir(save_dir)
    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'metric': metric,
        'state_dict': state_dict
    }, os.path.join(save_dir, tag + '.pth'))
