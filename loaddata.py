from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def DataFileWithKNN(rawdata, n_neighbors=2):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    print('n_neighbors',n_neighbors)

    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']
    filledData = rawdata.copy().drop('blood_type', axis=1)
    filledData.replace('N', 0, inplace=True)
    filledData.replace('N ', 0, inplace=True)
    filledData.replace('n', 0, inplace=True)
    filledData.replace('Y', 1, inplace=True)
    filledData.replace('y', 1, inplace=True)

    filledData.loc[:, 'blood'] = filledData.loc[:, days].apply(lambda x: x.sum(), 1)
    filledData.drop(columns=days, inplace=True)

    # filledData.fillna(filledData.loc[:, :"lowerbody"].mean(), inplace=True)
    filledData = filledData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
        'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites', 
        'chest', 'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']]
    
    # print(filledData.to_numpy())
    data_filled = imputer.fit_transform(filledData.to_numpy()[:,1:])
    filledData.iloc[:,1:] = data_filled

    return filledData

def preprocess(knn_neighbors=2, ifNorm=True, preSave=True, basePath:str="./data"):
    # nNullData = pd.read_csv(basePath + '/fullData.csv', index_col=0)
    # filledDataWithMean = pd.read_csv(basePath + '/filledDataWithMean.csv', index_col=0)
    # filledDataWithMiddle = pd.read_csv(basePath + '/filledDataWithMiddle.csv', index_col=0)
    if preSave:
        __save = np.load(basePath + '/nNull.npy', allow_pickle=True).item()
        __save1 = np.load(basePath + '/missing.npy', allow_pickle=True).item()
        __save2 = np.load(basePath + '/fillWithMean.npy', allow_pickle=True).item()
        __save3 = np.load(basePath + '/fillWithMiddle.npy', allow_pickle=True).item()
        __save4 = np.load(basePath + '/fillWithKNN.npy', allow_pickle=True).item()

        # label = np.load('data/nNullLabel.npy')
        # label1 = np.load('data/missingLabel.npy')
        # label2 = np.load('data/fillWithMeanLabel.npy')
        # label3 = np.load('data/fillWithMiddleLabel.npy')
        # label4 = np.load('data/fillWithKNNLabel.npy')
    else:
        rawdata = ReadRawData()
        # DataFileWithKNN(rawdata)
        missingData = SelectMissingData(rawdata)
        nNullData = SelectNoMissingData(rawdata)
        filledDataWithMean = DataFileWithMean(rawdata)
        filledDataWithMiddle = DataFileWithMedian(rawdata)
        filledDataWithKNN = DataFileWithKNN(rawdata, n_neighbors=knn_neighbors)

        __save, __save1, __save2, __save3, __save4 = {}, {}, {}, {}, {}
        tmp = np.zeros((2, nNullData.shape[1] - 4))

        tmp[0] = nNullData.loc[:, 'heart_rate':].max()
        tmp[1] = nNullData.loc[:, 'heart_rate':].min()
        data = pd.concat([nNullData.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1), nNullData.loc[:, 'upperbody':'c3']], axis=1).values
        label = nNullData.blood.copy().values.astype(int)
        __save['max_min'] = deepcopy(tmp)
        __save['data'] = data

        tmp[0] = missingData.loc[:, 'heart_rate':].max()
        tmp[1] = missingData.loc[:, 'heart_rate':].min()
        data1 = pd.concat([missingData.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), missingData.loc[:, 'upperbody':'c3']], axis=1).fillna(-1).values
        label1 = missingData.blood.copy().values.astype(int)
        __save1['max_min'] = deepcopy(tmp)
        __save1['data'] = data1

        tmp[0] = filledDataWithMean.loc[:, 'heart_rate':].max()
        tmp[1] = filledDataWithMean.loc[:, 'heart_rate':].min()
        data2 = pd.concat([filledDataWithMean.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), filledDataWithMean.loc[:, 'upperbody':'c3']], axis=1).values
        label2 = filledDataWithMean.blood.copy().values.astype(int)
        __save2['max_min'] = deepcopy(tmp)
        __save2['data'] = data2

        tmp[0] = filledDataWithMiddle.loc[:, 'heart_rate':].max()
        tmp[1] = filledDataWithMiddle.loc[:, 'heart_rate':].min()
        data3 = pd.concat([filledDataWithMiddle.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), filledDataWithMiddle.loc[:, 'upperbody':'c3']], axis=1).values
        label3 = filledDataWithMiddle.blood.copy().values.astype(int)
        __save3['max_min'] = deepcopy(tmp)
        __save3['data'] = data3

        tmp[0] = filledDataWithKNN.loc[:, 'heart_rate':].max()
        tmp[1] = filledDataWithKNN.loc[:, 'heart_rate':].min()
        data4 = pd.concat([filledDataWithKNN.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), filledDataWithMiddle.loc[:, 'upperbody':'c3']], axis=1).values
        label4 = filledDataWithKNN.blood.copy().values.astype(int)
        __save4['max_min'] = deepcopy(tmp)
        __save4['data'] = data4

        if ifNorm:
            label = (label - np.min(label)) / (np.max(label) - np.min(label))
            label1 = (label1 - np.min(label1)) / (np.max(label1) - np.min(label1))
            label2 = (label2 - np.min(label2)) / (np.max(label2) - np.min(label2))
            label3 = (label3 - np.min(label3)) / (np.max(label3) - np.min(label3))
            label4 = (label4 - np.min(label4)) / (np.max(label4) - np.min(label4))

        __save['label'] = label
        __save1['label'] = label1
        __save2['label'] = label2
        __save3['label'] = label3
        __save4['label'] = label4

        np.save('data/nNull.npy', __save, allow_pickle=True)
        np.save('data/missing.npy', __save1, allow_pickle=True)
        np.save('data/fillWithMean.npy', __save2, allow_pickle=True)
        np.save('data/fillWithMiddle.npy', __save3, allow_pickle=True)
        np.save('data/fillWithKNN.npy', __save4, allow_pickle=True)
    # return data, label, data1, label1, data2, label2, data3, label3, data4, label4
    return __save, __save1, __save2, __save3, __save4

def ReadRawData(basePath:str = './data'):
    df = pd.read_excel(basePath + '/all_hospital_v1_v2_for_article.xls')
    df.replace("B", "N", inplace=True)
    df[(df.blood_type.isna()) & (df.iloc[:, 23:].apply(
        lambda x: x.sum(), 1))].iloc[:, 22:]
    df = df[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate',
         'low_pressure', 'high_pressure', 'temperature',
         'hemoglobin', 'redcell', 'albumin', 'blood_type',
         'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
         'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15',
         'upperbody', 'lowerbody', 'penetrate', 'pleural',
         'ascites', 'chest', 'abdomen', 'pelvis', 'c1', 'c2', 'c3']
    ]
    return df


def SelectAllData(rawdata):
    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']

    allData = rawdata.drop('blood_type', axis=1).fillna(-1).copy()

    allData.replace('N', 0, inplace=True)
    allData.replace('N ', 0, inplace=True)
    allData.replace('n', 0, inplace=True)
    allData.replace('Y', 1, inplace=True)
    allData.replace('y', 1, inplace=True)

    allData.loc[:, 'blood'] = allData.loc[:, days].apply(lambda x: x.sum(), 1)
    allData.drop(columns=days, inplace=True)

    allData = allData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
        'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites', 'chest',
         'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']
         ]
    return allData


def SelectMissingData(rawdata):

    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']

    missingData = rawdata[rawdata.iloc[:, 4:].isnull().any(1)].drop(
        'blood_type', axis=1).copy()
    missingData.replace('N', 0, inplace=True)
    missingData.replace('N ', 0, inplace=True)
    missingData.replace('n', 0, inplace=True)
    missingData.replace('Y', 1, inplace=True)
    missingData.replace('y', 1, inplace=True)

    missingData.loc[:, 'blood'] = missingData.loc[:,
                                              days].apply(lambda x: x.sum(), 1)
    missingData.drop(columns=days, inplace=True)

    missingData = missingData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
         'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites', 'chest',
         'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']
    ]
    return missingData

def SelectNoMissingData(rawdata):

    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']

    nNullData = rawdata[~rawdata.iloc[:, 4:].isnull().any(1)].drop(
        'blood_type', axis=1).copy()
    nNullData.replace('N', 0, inplace=True)
    nNullData.replace('N ', 0, inplace=True)
    nNullData.replace('n', 0, inplace=True)
    nNullData.replace('Y', 1, inplace=True)
    nNullData.replace('y', 1, inplace=True)

    nNullData.loc[:, 'blood'] = nNullData.loc[:,
                                              days].apply(lambda x: x.sum(), 1)
    nNullData.drop(columns=days, inplace=True)

    nNullData = nNullData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
         'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites', 'chest',
         'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']
    ]
    return nNullData


def DataFileWithMean(rawdata):

    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']
    filledData = rawdata.copy().drop('blood_type', axis=1)
    filledData.replace('N', 0, inplace=True)
    filledData.replace('N ', 0, inplace=True)
    filledData.replace('n', 0, inplace=True)
    filledData.replace('Y', 1, inplace=True)
    filledData.replace('y', 1, inplace=True)

    filledData.loc[:, 'blood'] = filledData.loc[:, days].apply(lambda x: x.sum(), 1)
    filledData.drop(columns=days, inplace=True)

    filledData.fillna(filledData.loc[:, :"lowerbody"].mean(), inplace=True)
    filledData = filledData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
        'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites', 
        'chest', 'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']]

    return filledData


def DataFileWithMedian(rawdata):

    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']
    filledData = rawdata.copy().drop('blood_type', axis=1)
    filledData.replace('N', 0, inplace=True)
    filledData.replace('N ', 0, inplace=True)
    filledData.replace('n', 0, inplace=True)
    filledData.replace('Y', 1, inplace=True)
    filledData.replace('y', 1, inplace=True)

    filledData.loc[:, 'blood'] = filledData.loc[:, days].apply(lambda x: x.sum(), 1)
    filledData.drop(columns=days, inplace=True)

    filledData.fillna(filledData.loc[:, :"lowerbody"].median(), inplace=True)
    filledData = filledData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
        'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites', 
        'chest', 'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']]
    return filledData
