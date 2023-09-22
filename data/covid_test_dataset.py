import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform


class CovidTestDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, idx=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if idx is None:
            h5_name = "COVID_test.h5"
            print(f"Load: {h5_name}")
        else:
            h5_name = f"COVID_{idx}.h5"
            print(f"Load: {h5_name}")

        # self.is_test = True

        BaseDataset.__init__(self, opt)
        
        # self.covid_file = h5py.File(os.path.join(opt.dataroot, h5_name), 'r')
        self.covid_file = h5py.File(os.path.join(opt.val_dataroot, h5_name), 'r')

        if idx is None:
            test_db = self.covid_file['test']
            # train_db = self.covid_file
            self.dcm, self.label = self.build_pairs(test_db)
        else:
            train_db = self.covid_file['train']
            # train_db = self.covid_file
            self.dcm, self.label = self.build_pairs(train_db)

    def build_pairs(self, dataset):
        keys = dataset.keys()
        print('keys =', keys)
        dcm_arr = []
        label_arr = []

        for key in keys:
            print(f"build key:{key}")
            sys.stdout.flush()
            covid_img = dataset[f"{key}/data"][()]
            covid_label = dataset[f"{key}/label"][()]
            dcm_arr.append(covid_img)
            label_arr.append(covid_label)

        return dcm_arr, label_arr

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        img_data = self.dcm[index]
        img_label = self.label[index]

        img_data = Image.fromarray(img_data).convert('RGB')
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, img_data.size)
        img_data_transform = get_transform(self.opt, transform_params,  method=Image.NEAREST)

        img_data = img_data_transform(img_data)
        # print('-' * 10)
        # print('img_data type =', type(img_data))
        # print('img_data shape =', img_data.shape)
        return {'data': img_data, 'label': img_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dcm)
