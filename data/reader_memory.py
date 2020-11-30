import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from transforms.compose import Randomizable
from transforms.utils import apply_transform
from transforms.utility.array import ToTensor
from utils.misc import get_seed

import skimage.morphology as sm
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter


class MemDataset(Dataset, Randomizable):
    def __init__(self,
                 image_files,
                 seg_files,
                 cls_files=None,
                 image_transformed_files_lists=None,
                 weighted_contour_mask=False,
                 heatmap=False,
                 names=None,
                 transform=None,
                 transform_t=None,
                 seg_transform=None):

        self.image_files = image_files
        self.seg_files = seg_files
        self.cls_files = cls_files
        self.image_transformed_files_lists = image_transformed_files_lists
        self.weighted_contour_mask = weighted_contour_mask,
        self.heatmap = heatmap
        self.names = names
        self.transform = transform
        self.transform_t = transform_t
        self.seg_transform = seg_transform
        self.set_random_state(seed=get_seed())
        self.weighted = MaskMorphologic2D(dilation=10.0, sigma=5.0, magnitude=6.0, edge_enhance=weighted_contour_mask,
                                          hearMap=heatmap)
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.image_files)

    def randomize(self):
        self._seed = self.R.randint(np.iinfo(np.int32).max)

    def __getitem__(self, index):
        msk_out = []
        self.randomize()
        img = None
        msk = None
        cls = -1
        if self.image_files is not None:
            img = np.ascontiguousarray(self.image_files[index])
        if self.seg_files is not None:
            msk = np.ascontiguousarray(self.seg_files[index])
        if self.cls_files is not None:
            cls = np.ascontiguousarray(self.cls_files[index])

        if self.transform is not None and img is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            img = apply_transform(self.transform, img)

        if self.transform_t is not None and self.image_transformed_files_lists is not None:
            if isinstance(self.transform_t, Randomizable):
                self.transform_t.set_random_state(seed=self._seed)
            for img_t_file in self.image_transformed_files_lists:
                img_t = np.ascontiguousarray(img_t_file[index])
                img_t = apply_transform(self.transform_t, img_t)
                img = np.concatenate((img, img_t), axis=0)

        if self.seg_transform is not None and msk is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)
            msk = apply_transform(self.seg_transform, msk)
            msk_tensor = self.totensor(msk)
            msk_out.append(msk_tensor)

        if (self.weighted_contour_mask == True or self.heatmap == True) and msk is not None:
            msk_weight = self.weighted(msk)
            msk_weight_tensor = self.totensor(msk_weight)
            msk_out.append(msk_weight_tensor)

        if self.names is not None:
            name = self.names[index]
        else:
            name = 'unknown'

        if msk is not None and self.cls_files is not None:
            return img, msk_out, cls, name
        elif msk is not None and self.cls_files is None:
            return img, msk_out, name
        elif msk is None and self.cls_files is not None:
            return img, cls, name
        else:
            return img, name

class MaskMorphologic2D():
    def __init__(self, dilation=5, sigma=5.0, magnitude=4.0, edge_enhance=False, hearMap=False):
        self.dilation = dilation
        self.sigma = sigma
        self.mag = magnitude
        self.edge_enhance = edge_enhance
        self.heatmap = hearMap

    def EdgeEnhance(self, mask):
        mask_cp = mask.copy()
        dx = np.abs(mask_cp[:, :, 1:] - mask_cp[:, :, :-1])
        dy = np.abs(mask_cp[:, 1:, :] - mask_cp[:, :-1, :])
        dx = dx[:, 1:, :]
        dy = dy[:, :, 1:]
        dxy_ = np.abs(dx + dy)
        dxy = np.zeros(mask.shape, dtype='int16')
        dxy[:, 1:, 1:] = dxy_
        dxy = binary_dilation(dxy[0], sm.square(self.dilation), iterations=1)
        dxy = dxy[np.newaxis, :].astype(np.float)
        dxy[dxy > 0] = self.mag
        dxy = gaussian_filter(dxy, sigma=self.sigma)
        dxy += 1.0
        return dxy

    def HeatMap(self, mask):
        _, px, py = np.where(mask > 0)
        if len(px) == 0 or len(py) == 0:
            return mask
        meanx = int(np.round(px.mean()))
        meany = int(np.round(py.mean()))
        dxy = np.zeros(mask.shape, dtype='int16')
        dxy[:, meanx, meany] = 1
        dxy = binary_dilation(dxy[0], sm.disk(self.dilation), iterations=1)
        dxy = dxy[np.newaxis, :].astype(np.float)
        dxy[dxy > 0] = 1

    def __call__(self, mask):
        if self.edge_enhance:
            return self.EdgeEnhance(mask)
        elif self.heatmap:
            return self.HeatMap(mask)
        else:
            return mask

