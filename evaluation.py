import torch
import numpy as np
import argparse
import cv2
import os
import csv
import json

from tqdm import tqdm

from networks.nets.deeplab_v3 import DeepLab
from networks.utils import build_network


def draw_contour(img, mask, gt):
    contour_map = np.zeros([3, img.shape[0], img.shape[1]], dtype=np.uint8)
    img = img[np.newaxis, :]
    contour_map[0, :, :] = img[0, :, :]
    contour_map[1, :, :] = img[0, :, :]
    contour_map[2, :, :] = img[0, :, :]

    mask_colors = [255, 0, 0] # red
    gt_colors = [0, 255, 0] # green

    mask = mask.astype(np.uint8)
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in mask_contours:
        contour_map[0, c[:, 0, 1], c[:, 0, 0]] = int(mask_colors[0])
        contour_map[1, c[:, 0, 1], c[:, 0, 0]] = int(mask_colors[1])
        contour_map[2, c[:, 0, 1], c[:, 0, 0]] = int(mask_colors[2])

    # gt = gt.astype(np.uint8)
    # gt_contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for c in gt_contours:
    #     contour_map[0, c[:, 0, 1], c[:, 0, 0]] = int(gt_colors[0])
    #     contour_map[1, c[:, 0, 1], c[:, 0, 0]] = int(gt_colors[1])
    #     contour_map[2, c[:, 0, 1], c[:, 0, 0]] = int(gt_colors[2])

    return contour_map


def restore_original_img(ori_img_path, mask, bbox):
    img = cv2.imread(ori_img_path)
    img = np.transpose(img, (2, 0, 1))

    contour_map = np.zeros([img.shape[0], img.shape[1], img.shape[2]], dtype=np.uint8)
    contour_map[0, :, :] = img[0, :, :]
    contour_map[1, :, :] = img[1, :, :]
    contour_map[2, :, :] = img[2, :, :]

    mask_colors = [255, 0, 0]  # red
    gt_colors = [0, 255, 0]  # green

    mask = mask.astype(np.uint8)
    mask_map = np.zeros([img.shape[1], img.shape[2]], dtype=np.uint8)
    mask_map[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = mask

    mask_contours, _ = cv2.findContours(mask_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in mask_contours:
        contour_map[0, c[:, 0, 1], c[:, 0, 0]] = int(mask_colors[0])
        contour_map[1, c[:, 0, 1], c[:, 0, 0]] = int(mask_colors[1])
        contour_map[2, c[:, 0, 1], c[:, 0, 0]] = int(mask_colors[2])

    # gt = gt.astype(np.uint8)
    # gt_contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for c in gt_contours:
    #     contour_map[0, c[:, 0, 1], c[:, 0, 0]] = int(gt_colors[0])
    #     contour_map[1, c[:, 0, 1], c[:, 0, 0]] = int(gt_colors[1])
    #     contour_map[2, c[:, 0, 1], c[:, 0, 0]] = int(gt_colors[2])

    return contour_map


def tta_predict(model, data):
    '''
    使用不同变换下的样例，得出多个预测结果进行加权计算
    :param model:
    :param data:
    :return:
    '''
    data = data.type(torch.FloatTensor)
    val_data_list = []
    prob_map_list = []

    # aug 8 times
    for i in range(data.shape[0]):
        for j in range(4):
            rot_data = torch.rot90(torch.clone(data[i]), j, (1, 2))
            flip_data = torch.flip(torch.clone(rot_data), (1,))
            val_data_list.append(rot_data)
            val_data_list.append(flip_data)

    val_volume = torch.stack(val_data_list, 0)
    # val_volume = val_volume.unsqueeze(1)
    val_volume = val_volume.cuda()

    val_outputs = model(val_volume)

    val_outputs = val_outputs.sigmoid().cpu()
    for i in range(data.shape[0]):
        val_outputs_list = []
        for j in range(8):
            rot_data = torch.rot90(val_outputs[i * 8 + j, 0], 4 - j // 2, (0, 1))
            if j % 2 == 0:
                val_outputs_list.append(rot_data)
            else:
                flip_data = torch.flip(torch.clone(rot_data), (1 - (((j + 1) // 2) % 2),))
                val_outputs_list.append(flip_data)
        val_outputs_stack = torch.stack(val_outputs_list, 0)
        pro_map = torch.mean(val_outputs_stack, 0)
        prob_map_list.append(pro_map)

    prob_maps = torch.stack(prob_map_list, 0).unsqueeze(1)
    prob_maps_np = prob_maps.cpu().detach().numpy().astype(np.float)
    pred_mask_np = (prob_maps_np >= 0.5).astype(np.uint8)
    return pred_mask_np


def normalize(data):
    data = data.astype(np.float)
    nor_data = (data - np.mean(data)) / np.std(data)
    nor_data = nor_data.astype(np.float)
    return nor_data


def main(args):

    model = DeepLab(backbone='resnet', output_stride=16, num_classes=1, sync_bn=True, freeze_bn=False, dcn=False, g_in_dim=1)
    model, best_metric, best_metric_epoch = build_network(model, args.checkpoint, args.snapshot)
    model.eval()

    # --------------------------------
    # load data

    # Method 1
    data_list = os.listdir(args.dataset_path)

    # Method 2
    # with open(r'./data/train.csv', 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     rows = [row for row in reader]
    # rows = rows[1:]
    #
    # data_list = []
    # for i in rows:
    #     if int(i[1]) in [9]:
    #         data_list.append(i[0])

    # return data list, include file name
    # --------------------------------

    count = 0

    for i in data_list:
        count+=1
        print(count)
        name_id = i.split('.')[0]
        # name_id = '3034'

        # img_path = os.path.join(args.dataset_path, i+'.PNG')
        img_path = os.path.join(args.dataset_path, name_id+'.png')
        # img_path = r'/data/sdd/muqiu.qyx/Dataset/test/3034.jpg'
        ori_img = cv2.imread(img_path, 0)
        ori_img_shape = ori_img.shape

        img = cv2.resize(ori_img, (512, 512), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=2)
        img = torch.tensor(np.transpose(normalize(img), (2, 0, 1)))
        img = img.unsqueeze(0)

        mask_pre = tta_predict(model, img)

        mask_pre[mask_pre > 0] = 255
        mask_pre = np.squeeze(mask_pre)
        mask_pre = cv2.resize(mask_pre, (ori_img_shape[1], ori_img_shape[0]), interpolation=cv2.INTER_AREA)

        # from PIL import Image
        # im = Image.fromarray(mask_pre)
        # im.save(os.path.join(args.save_results_path, name_id+'_mask.png'))

        # mask_gt = cv2.imread(os.path.join(r'/home/user/WorkSpace/DataSet/TNSCUI2020/mask', i+'.PNG'), 0)
        ori_img_path = os.path.join(r'/home/user/WorkSpace/DataSet/test', name_id.split('_cut')[0]+'.jpg')
        json_path = os.path.join(r'/home/user/WorkSpace/DataSet/test_cut_json', name_id+'.json')

        with open(json_path, 'r') as load_f:
            load_dict = json.load(load_f)
        bbox = [int(load_dict['x']), int(load_dict['y']), int(load_dict['w']), int(load_dict['h'])]

        result = restore_original_img(ori_img_path, mask_pre, bbox)

        # result = draw_contour(ori_img, mask_pre, None)
        result = np.ascontiguousarray(result.transpose(1, 2, 0))
        from PIL import Image
        im = Image.fromarray(result)
        im.save(os.path.join(args.save_results_path, name_id+'.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='code for evaluation')
    parser.add_argument('--checkpoint', default='./checkpoint/', type=str, help='checkpoint path')
    parser.add_argument('--snapshot', default='best_metric_model.pth', type=str, help='checkpoint file')

    parser.add_argument('--dataset_path', default=r'/home/user/WorkSpace/DataSet/test_cut_image', type=str, help='dataset path')
    parser.add_argument('--save_results_path', default=r'/home/user/WorkSpace/DataSet/test_cut_save_new', type=str, help='save results path')
    # /data/sdd/muqiu.qyx/segmentation_project/result
    args = parser.parse_args()
    main(args)