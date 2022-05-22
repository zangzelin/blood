# from source import Source
import load_data_f.source as Source
from sklearn.datasets import load_digits
import torchvision.datasets as datasets
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from PIL import Image
import os
# import torchtext
# import scanpy as sc
import scipy
from sklearn.decomposition import PCA
import pandas as pd
from loaddata import ReadRawData, SelectNoMissingData, DataFileWithMean, DataFileWithMedian, preprocess
from sklearn.model_selection import train_test_split, StratifiedKFold


class BlodNoMissingDataModule(Source.Source):

    def _LoadData(self):
        # rawdata = ReadRawData()
        # nNullData = SelectNoMissingData(rawdata)
        # data = pd.concat([nNullData.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
        #     x - np.min(x)) / (np.max(x) - np.min(x))), nNullData.loc[:, 'upperbody':'c3']], axis=1).values
        # label = nNullData.blood.copy().values.astype(int)
        # # label[label > 0] = 1
        # label[label > 0.5] = 1
        # label[label <= 0.5] = 0

        # kf = KFold(n_splits=10)
        # print(data, '\n', label)

        (
            nNull, missing, fillWithMean, fillWithMiddle, fillWithKNN,
        ) = preprocess(knn_neighbors=2, ifNorm=True, preSave=True)

        if self.fill_set == 'nNull':
            data, label = nNull['data'], nNull['label']
            # print("data, label = nNull['data'], nNull['label']")
        elif self.fill_set == 'fillWithMiddle':
            data, label = fillWithMiddle['data'], fillWithMiddle['label']
            # print("data, label = fillWithMiddle['data'], fillWithMiddle['label']")
        elif self.fill_set == 'fillWithKNN':
            data, label = fillWithKNN['data'], fillWithKNN['label']
            # print("data, label = fillWithKNN['data'], fillWithKNN['label']")
        elif self.fill_set == 'fillWithMean':
            data, label = fillWithMean['data'], fillWithMean['label']
            # print("data, label = fillWithMean['data'], fillWithMean['label']")
        tl = "Not Null"
        # data, label = fillWithMean['data'], fillWithMean['label']
        # tl = "Fill With Mean"
        # data, label = fillWithMiddle['data'], fillWithMiddle['label']
        # tl = "Fill With Middle"

        label[label > 0] = 1

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, stratify=label, random_state=1)
        skf = StratifiedKFold(n_splits=10, random_state=2022, shuffle=True)
        acc, auc = {}, {}
        # for clf_key in clfs.keys():
        #     print('the classifier is : {}'.format(clf_key))
        #     acc[clf_key], auc[clf_key] = [], []
        for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
            if i == self.foldindex:
                # print(self.foldindex, test_index)
                train_X, train_y = X_train[train_index], y_train[train_index]
                val_X, val_y = X_train[val_index], y_train[val_index]
                data = np.array(train_X)
                label = np.array(train_y)
                dataval = np.array(val_X)
                labelval = np.array(val_y)

        # digit = load_digits()
        self.data = torch.tensor(data).float()
        self.dataval = torch.tensor(dataval).float()
        self.datatest = torch.tensor(X_test).float()

        self.label = torch.tensor(np.array(label))
        self.labelval = torch.tensor(np.array(labelval))
        self.labeltest = torch.tensor(np.array(y_test))

        self.inputdim = self.data[0].shape
        self.same_sigma = False
        # print('shape = ', self.data.shape)
        self.label_str = [label]


class BlodAllDataModule(Source.Source):

    def _LoadData(self):
        # print('load DigitsDataModule')

        # if not self.train:
        #     random_state = self.random_state + 1
        datapei_raw = pd.read_excel('data/all_hospital_v1_v2_for_article.xls')
        datapei_raw = datapei_raw.replace('N', 0)
        datapei_raw = datapei_raw.replace('N ', 0)
        datapei_raw = datapei_raw.replace('Y', 1)
        datapei_raw = datapei_raw.replace('Y ', 1)
        datapei_raw = datapei_raw.replace('y', 1)
        datapei_raw = datapei_raw.replace('B', 0)

        # datapei_raw = datapei_raw[
        #     datapei_raw['hospital'].str.contains('zheer')
        #     ]
        labelname = ['d' + str(i + 1) for i in range(14)]
        label = np.log(1 + datapei_raw[labelname].sum(axis=1).to_numpy())

        data = datapei_raw.drop(['blood_type', 'hospital'] + labelname, axis=1)
        data = data.fillna(data.median()).to_numpy().astype(np.float32)

        print(data)

        # digit = load_digits()
        self.data = torch.tensor(data).float()
        self.label = np.array(label)
        self.inputdim = self.data[0].shape
        self.same_sigma = False
        print('shape = ', self.data.shape)
        self.label_str = [label]
