import os
import cv2
import csv
import numpy as np
from tqdm import tqdm


class SEED2020DataList():
    def __init__(self, dataset_path=None, csvfile_path=None, image_size=(512, 512), train_branch=None, val_branch=None):

        super(SEED2020DataList, self).__init__()
        self.dataset_path = dataset_path
        self.csvfile_path = csvfile_path

        self.image_size = image_size
        self.train_branch = train_branch
        self.val_branch = val_branch

        self.list_csv = self.read_csv()

    def read_csv(self):
        with open(self.csvfile_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        rows = rows[1:]
        return rows

    def get_train_list(self):

        train_data_dict = {}
        train_mask_dict = {}
        train_cate_dict = {}

        for i in tqdm(self.list_csv):
            img_path = os.path.join(self.dataset_path, i[0]+'.jpg')
            mask_path = os.path.join(self.dataset_path, i[0]+'_mask.jpg')

            cate = 1

            if self.train_branch is not None and int(i[1]) in self.train_branch:
                img = cv2.imread(img_path)
                msk = self.adjust_mask(cv2.imread(mask_path, 0))

                if self.image_size is not None:
                    img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, self.image_size, interpolation=cv2.INTER_AREA)

                train_data_dict[i[0]] = np.transpose(self.normalize(img), (2, 0, 1))
                train_mask_dict[i[0]] = self.binary(msk)
                train_cate_dict[i[0]] = cate

            # if len(list(train_data_dict)) > 200:
            #     break

        return train_data_dict, train_mask_dict, train_cate_dict

    def get_val_list(self):
        val_data_dict = {}
        val_mask_dict = {}
        val_cate_dict = {}

        for i in tqdm(self.list_csv):
            img_path = os.path.join(self.dataset_path, i[0] + '.jpg')
            mask_path = os.path.join(self.dataset_path, i[0] + '_mask.jpg')

            cate = 1

            if self.val_branch is not None and int(i[1]) in self.val_branch:
                img = cv2.imread(img_path)
                msk = self.adjust_mask(cv2.imread(mask_path, 0))

                if self.image_size is not None:
                    img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, self.image_size, interpolation=cv2.INTER_AREA)

                val_data_dict[i[0]] = np.transpose(self.normalize(img), (2, 0, 1))
                val_mask_dict[i[0]] = self.binary(msk)
                val_cate_dict[i[0]] = cate

            # if len(list(val_data_dict)) > 200:
            #     break

        return val_data_dict, val_mask_dict, val_cate_dict

    def adjust_mask(self, mask):
        mask[mask < 10] = 0
        mask[mask > 0] = 1
        return mask.astype(np.float)

    def normalize(self, data):
        data = data.astype(np.float)
        nor_data = (data - np.mean(data)) / np.std(data)
        nor_data = nor_data.astype(np.float)
        return nor_data

    def binary(self, mask):
        mask[mask > 0] = 1
        return mask.astype(np.float)


class TNSCUI2020DataList():
    def __init__(self, dataset_path=None, csvfile_path=None, image_size=(512, 512), train_branch=None, val_branch=None):
        super(TNSCUI2020DataList, self).__init__()
        self.dataset_path = dataset_path
        self.csvfile_path = csvfile_path

        self.image_size = image_size
        self.train_branch = train_branch
        self.val_branch = val_branch

        self.list_csv = self.read_csv()

    def read_csv(self):
        with open(self.csvfile_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        rows = rows[1:]
        return rows

    def get_train_list(self):
        train_data_dict = {}
        train_mask_dict = {}
        train_cate_dict = {}

        for i in tqdm(self.list_csv):
            img_path = os.path.join(self.dataset_path, 'image', i[0]+'.PNG')
            mask_path = os.path.join(self.dataset_path, 'mask', i[0]+'.PNG')
            cate = 1

            if self.train_branch is not None and int(i[1]) in self.train_branch:
                img = cv2.imread(img_path, 0)
                msk = cv2.imread(mask_path, 0)

                if self.image_size is not None:
                    img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, self.image_size, interpolation=cv2.INTER_AREA)

                train_data_dict[i[0]] = self.normalize(img)
                train_mask_dict[i[0]] = self.binary(msk)
                train_cate_dict[i[0]] = cate

            # if len(list(train_data_dict)) > 200:
            #     break

        return train_data_dict, train_mask_dict, train_cate_dict

    def get_val_list(self):
        val_data_dict = {}
        val_mask_dict = {}
        val_cate_dict = {}

        for i in tqdm(self.list_csv):
            img_path = os.path.join(self.dataset_path, 'image', i[0]+'.PNG')
            mask_path = os.path.join(self.dataset_path, 'mask', i[0]+'.PNG')
            cate = 1
            if self.val_branch is not None and int(i[1]) in self.val_branch:
                img = cv2.imread(img_path, 0)
                msk = cv2.imread(mask_path, 0)

                if self.image_size is not None:
                    img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                    msk = cv2.resize(msk, self.image_size, interpolation=cv2.INTER_AREA)
                val_data_dict[i[0]] = self.normalize(img)
                val_mask_dict[i[0]] = self.binary(msk)
                val_cate_dict[i[0]] = cate

            # if len(list(val_data_dict)) > 50:
            #     break

        return val_data_dict, val_mask_dict, val_cate_dict

    def normalize(self, data):
        data = data.astype(np.float)
        nor_data = (data - np.mean(data)) / np.std(data)
        nor_data = nor_data.astype(np.float)
        return nor_data

    def binary(self, mask):
        mask[mask > 0] = 1
        return mask.astype(np.float)



if __name__ == '__main__':
    dataloader = TNSCUI2020DataList(dataset_path=r'/home/user/WorkSpace/DataSet/TNSCUI2020',
                                    csvfile_path=r'train.csv',
                                    image_size=(512, 512),
                                    train_branch=[0],
                                    val_branch=[9])

    train_data_list = dataloader.get_train_list()
    val_data_list = dataloader.get_val_list()