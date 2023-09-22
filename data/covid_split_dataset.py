import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.covid_dataset import CovidDataset


class CovidSplitDataset(BaseDataset):

    def __init__(self, opt):

        self.split_db = []
        for i in range(3):
            self.split_db.append(CovidDataset(opt, i))

        # print('split_db type =', type(self.split_db))
        # print('split_db length =', len(self.split_db))
        # print('split_db[0] length =', len(self.split_db[0]))
    def __getitem__(self, index):
        count = 0
        result = {}

        # print('----------------------------------index begin----------------------------')
        for k, v in enumerate(self.split_db):
            # print('-------------for database begin-----------------')
            database = v

            if index >= len(database):
                # print('len(database) = ',len(database))
                index = index % len(database)

            index_value = database[index]
            result['data_' + str(k)] = index_value['data']
            result['label_' + str(k)] = index_value['label']

        return result

    def __len__(self):
        """Return the total number of images in the dataset."""
        length = 0
        for i in self.split_db:
            if len(i) > length:
                length = len(i)


        return length

