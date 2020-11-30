import torch
import math
import numpy as np
import argparse

from torch.utils.data import DataLoader

from networks.nets.deeplab_v3 import DeepLab
from networks.utils import build_network
from losses.dice import DiceLoss
from utils.metrics import AverageMeter, MetricAIC
from utils.file_io import save_model

from data.dataloader import SEED2020DataList
from data.reader_memory import MemDataset
from transforms.compose import Compose
from transforms.utility.array import AddChannel, ToTensor
from transforms.spatial.array import RandRotate90, RandFlip, Rand2DElastic
from transforms.intensity.array import RandGaussianNoise, RandScaleIntensity, RandShiftIntensity


def main(args):
    lr = 0.006
    lr_p = 0.6
    batch_size = 24
    num_workers = 32

    train_branch = [4, 1, 2, 3]
    val_branch = [0]

    usDataList = SEED2020DataList(dataset_path=args.dataset_path,
                                  csvfile_path=args.csvfile_path,
                                  train_branch=train_branch, val_branch=val_branch)

    images_list, mask_list, _ = usDataList.get_train_list()
    images_val_list, mask_val_list, _ = usDataList.get_val_list()
    print("Train data set: ", len(images_list))
    print("Train data mask set: ", len(mask_list))
    print("Val data set: ", len(images_val_list))
    print('Val data mask set: ', len(mask_val_list))

    names_train_list = list(images_list.keys())
    images_train_list = list(images_list.values())
    label_train_list = list(mask_list.values())
    names_valid_list = list(images_val_list.keys())
    images_valid_list = list(images_val_list.values())
    label_valid_list = list(mask_val_list.values())

    prob = 0.5
    train_imtrans = Compose(
        [
            # AddChannel(), # add channel: (w, h) -> (c, w, h)
            RandRotate90(prob=prob, spatial_axes=(0, 1)),
            RandFlip(prob=prob, spatial_axis=(0, 1)),
            Rand2DElastic(prob=prob,
                          spacing=(100, 100),
                          magnitude_range=(10, 20),
                          rotate_range=(np.pi / 10.0,),
                          scale_range=(0.05, 0.1),
                          translate_range=(0, 5),
                          padding_mode='border'),
            RandScaleIntensity(prob=prob, factors=0.2),
            RandShiftIntensity(prob=prob, offsets=0.2),
            RandGaussianNoise(prob=prob, mean=0.0, std=0.4),
            ToTensor()
        ]
    )
    train_segtrans = Compose(
        [
            AddChannel(),
            RandRotate90(prob=prob, spatial_axes=(0, 1)),
            RandFlip(prob=prob, spatial_axis=(0, 1)),
            Rand2DElastic(prob=prob,
                          spacing=(100, 100),
                          magnitude_range=(10, 20),
                          rotate_range=(np.pi / 10.0,),
                          scale_range=(0.05, 0.1),
                          translate_range=(0, 5),
                          padding_mode='border'),
        ]
    )
    val_imtrans = Compose(
        [
            # AddChannel(),
            ToTensor()
        ]
    )
    val_segtrans = Compose([AddChannel()])

    train_ds = MemDataset(images_train_list, label_train_list, names=names_train_list,
                          weighted_contour_mask=False, transform=train_imtrans,
                          seg_transform=train_segtrans)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True,
                              pin_memory=torch.cuda.is_available())

    val_ds = MemDataset(images_valid_list, label_valid_list, names=names_valid_list,
                        weighted_contour_mask=False, transform=val_imtrans,
                        seg_transform=val_segtrans)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True,
                            pin_memory=torch.cuda.is_available())

    model = DeepLab(backbone='resnet', output_stride=16, num_classes=1, sync_bn=True, freeze_bn=False, dcn=False)
    model, best_metric, best_metric_epoch = build_network(model, args.checkpoint, args.snapshot)

    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=lr_p, verbose=True)

    val_interval = 1
    metric_values = list()

    for epoch in range(best_metric_epoch, best_metric_epoch+args.epochs):
        print('-' * 10)
        print(f"epoch {epoch + 1}")

        segloss = AverageMeter()

        model.train()

        step = 0
        for batch_data, batch_mask, batch_name in train_loader:
            step += 1
            inputs = batch_data.cuda()
            labels = batch_mask[0].cuda()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            segloss.update(loss.item())
            epoch_len = math.ceil(len(train_ds) / train_loader.batch_size)

            print(f"{step}/{epoch_len}, learning_rate: {optimizer.param_groups[0]['lr']}, train_loss: {loss.item():.4f}")

        print(f"epoch {epoch+1} average loss: {segloss.avg:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()

            with torch.no_grad():
                metric_sum = 0.0
                metric_count = 0

                for val_data, val_mask, val_name in val_loader:
                    val_data = val_data.type(torch.FloatTensor)
                    val_images, val_labels = val_data.cuda(), val_mask[0].cuda()

                    val_outputs = model(val_images)

                    tmp_result = (val_outputs.sigmoid() >= 0.5).cpu().numpy().astype(np.uint8)
                    tmp_result[tmp_result > 0] = 1
                    tmp_mask = val_mask[0].numpy().squeeze()
                    tmp_mask[tmp_mask > 0] = 1
                    tmp_result = tmp_result.astype(np.uint8)
                    tmp_mask = tmp_mask.astype(np.uint8)

                    for b in range(tmp_result.shape[0]):
                        met = MetricAIC(y_true=tmp_mask[b], y_pred=tmp_result[b])
                        metric_sum += met.dice_coef()
                        metric_count += 1
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    save_model(model, best_metric, best_metric_epoch, args.checkpoint)
                    print('saved new best metric model')
                print(
                    'current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='basic seg algorithm')
    parser.add_argument('--checkpoint', default='./checkpoint/', type=str, help='checkpoint path')
    parser.add_argument('--snapshot', default='best_metric_model.pth', type=str, help='checkpoint file')

    parser.add_argument('--dataset_path', default=r'/data/sdd/muqiu.qyx/Dataset/train', type=str, help='dataset path')
    parser.add_argument('--csvfile_path', default=r'./data/train.csv', type=str, help='csvfile path')

    parser.add_argument('--epochs', default=1000, type=int, help='epochs')
    args = parser.parse_args()
    main(args)