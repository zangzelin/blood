from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def preprocess(knn_neighbors=2, ifNorm=True, preSave=True, hos: str = "all", basePath: str = "./data"):
    # nNullData = pd.read_csv(basePath + '/fullData.csv', index_col=0)
    # filledDataWithMean = pd.read_csv(basePath + '/filledDataWithMean.csv', index_col=0)
    # filledDataWithMiddle = pd.read_csv(basePath + '/filledDataWithMiddle.csv', index_col=0)
    if preSave:
        __save = np.load(basePath + f'/nNull_{hos}.npy', allow_pickle=True).item()
        __save1 = np.load(basePath + f'/missing_{hos}.npy', allow_pickle=True).item()
        __save2 = np.load(basePath + f'/fillWithMean_{hos}.npy', allow_pickle=True).item()
        __save3 = np.load(basePath + f'/fillWithMiddle_{hos}.npy', allow_pickle=True).item()
        __save4 = np.load(basePath + f'/fillWithKNN_{hos}.npy', allow_pickle=True).item()
    else:
        rawdata = ReadRawData()
        missingData = SelectMissingData(rawdata)
        nNullData = SelectNoMissingData(rawdata)
        filledDataWithMean = DataFileWithMean(rawdata)
        filledDataWithMiddle = DataFileWithMedian(rawdata)
        filledDataWithKNN = DataFileWithKNN(rawdata, n_neighbors=knn_neighbors)

        print(missingData.shape, nNullData.shape, filledDataWithMean.shape, filledDataWithMiddle.shape)

        __save, __save1, __save2, __save3, __save4 = {}, {}, {}, {}, {}
        tmp = np.zeros((2, nNullData.shape[1] - 4))

        tmp[0] = nNullData.loc[:, 'heart_rate':].max()
        tmp[1] = nNullData.loc[:, 'heart_rate':].min()
        data = pd.concat([nNullData.loc[:, 'hospital'], nNullData.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), nNullData.loc[:, 'upperbody':'c3']], axis=1)
        data = data[data.hospital == hos].iloc[:, 1:].values if hos != 'all' else data.iloc[:, 1:].values
#         print(data.columns)
#         data = data.values
        label = nNullData.blood.copy().values.astype(int)
        __save['max_min'] = deepcopy(tmp)
        __save['data'] = data

        tmp[0] = missingData.loc[:, 'heart_rate':].max()
        tmp[1] = missingData.loc[:, 'heart_rate':].min()
        data1 = pd.concat([missingData.loc[:, 'hospital'], missingData.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), missingData.loc[:, 'upperbody':'c3']], axis=1).fillna(-1)
        data1 = data1[data1.hospital == hos].iloc[:, 1:].values if hos != 'all' else data1.iloc[:, 1:].values
        label1 = missingData.blood.copy().values.astype(int)
        __save1['max_min'] = deepcopy(tmp)
        __save1['data'] = data1

        tmp[0] = filledDataWithMean.loc[:, 'heart_rate':].max()
        tmp[1] = filledDataWithMean.loc[:, 'heart_rate':].min()
        data2 = pd.concat([filledDataWithMean.loc[:, 'hospital'], filledDataWithMean.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), filledDataWithMean.loc[:, 'upperbody':'c3']], axis=1)
        data2 = data2[data2.hospital == hos].iloc[:, 1:].values if hos != 'all' else data2.iloc[:, 1:].values
        label2 = filledDataWithMean.blood.copy().values.astype(int)
        __save2['max_min'] = deepcopy(tmp)
        __save2['data'] = data2

        tmp[0] = filledDataWithMiddle.loc[:, 'heart_rate':].max()
        tmp[1] = filledDataWithMiddle.loc[:, 'heart_rate':].min()
        data3 = pd.concat([filledDataWithMiddle.loc[:, 'hospital'], filledDataWithMiddle.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), filledDataWithMiddle.loc[:, 'upperbody':'c3']], axis=1)
        data3 = data3[data3.hospital == hos].iloc[:, 1:].values if hos != 'all' else data3.iloc[:, 1:].values
        label3 = filledDataWithMiddle.blood.copy().values.astype(int)
        __save3['max_min'] = deepcopy(tmp)
        __save3['data'] = data3

        tmp[0] = filledDataWithKNN.loc[:, 'heart_rate':].max()
        tmp[1] = filledDataWithKNN.loc[:, 'heart_rate':].min()
        data4 = pd.concat([filledDataWithKNN.loc[:, 'hospital'], filledDataWithKNN.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
            x - np.min(x)) / (np.max(x) - np.min(x))), filledDataWithKNN.loc[:, 'upperbody':'c3']], axis=1)
        data4 = data4[data4.hospital == hos].iloc[:, 1:].values if hos != 'all' else data4.iloc[:, 1:].values
        label4 = filledDataWithKNN.blood.copy().values.astype(int)
        __save4['max_min'] = deepcopy(tmp)
        __save4['data'] = data4

        if ifNorm:
            try:
                label = (label - np.min(label)) / (np.max(label) - np.min(label))
            except ValueError as e:
                print(repr(e))
            try:
                label1 = (label1 - np.min(label1)) / (np.max(label1) - np.min(label1))
            except ValueError as e:
                print(repr(e))
            try:
                label2 = (label2 - np.min(label2)) / (np.max(label2) - np.min(label2))
            except ValueError as e:
                print(repr(e))
            try:
                label3 = (label3 - np.min(label3)) / (np.max(label3) - np.min(label3))
            except ValueError as e:
                print(repr(e))
            try:
                label4 = (label4 - np.min(label4)) / (np.max(label4) - np.min(label4))
            except ValueError as e:
                print(repr(e))
            
        __save['label'] = label
        __save1['label'] = label1
        __save2['label'] = label2
        __save3['label'] = label3
        __save4['label'] = label4

        np.save(f'data/nNull_{hos}.npy', __save, allow_pickle=True)
        np.save(f'data/missing_{hos}.npy', __save1, allow_pickle=True)
        np.save(f'data/fillWithMean_{hos}.npy', __save2, allow_pickle=True)
        np.save(f'data/fillWithMiddle_{hos}.npy', __save3, allow_pickle=True)
        np.save(f'data/fillWithKNN_{hos}.npy', __save4, allow_pickle=True)
    # return data, label, data1, label1, data2, label2, data3, label3, data4, label4
    return __save, __save1, __save2, __save3, __save4


def ReadRawData(basePath: str = './data'):
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

    df.replace('N', 0, inplace=True)
    df.replace('N ', 0, inplace=True)
    df.replace('n', 0, inplace=True)
    df.replace('Y', 1, inplace=True)
    df.replace('y', 1, inplace=True)

    df.replace('wenling', 'WLing', inplace=True)
    df.replace('shangyu', 'SYu', inplace=True)
    df.replace('shiyi', 'SYi', inplace=True)
    df.replace('dongyang', 'DYang', inplace=True)
    df.replace('xiaoshan', 'XShan', inplace=True)
    df.replace('enze', 'EZe', inplace=True)
    df.replace('yongkang', 'YKang', inplace=True)
    df.replace('haining', 'HNing', inplace=True)
    df.replace('yuyao', 'YYao', inplace=True)
    df.replace('xinchang', 'XChang', inplace=True)
    df.replace('shaoyifu', 'SYiFu', inplace=True)
    df.replace('shaoyifu2018L1', 'SYiFu', inplace=True)
    df.replace('shaoyifu2018L2', 'SYiFu', inplace=True)
    df.replace('shaoyifu2020L1', 'SYiFu', inplace=True)
    df.replace('shaoyifu2020L2', 'SYiFu', inplace=True)
    df.replace('zheer2018', 'ZEr', inplace=True)
    df.replace('zheer2019', 'ZEr', inplace=True)
    df.replace('zheer2020', 'ZEr', inplace=True)
    df.replace('shiyi2020', 'ZEr', inplace=True)
    df.replace('wenling_explode', 'ZEr', inplace=True)
        
    return df


def SelectAllData(rawdata):
    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']
    
    allData = rawdata.drop('blood_type', axis=1).fillna(-1).copy()

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

    missingData.loc[:, 'blood'] = missingData.loc[:, days].apply(lambda x: x.sum(), 1)
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
    
    nNullData.loc[:, 'blood'] = nNullData.loc[:, days].apply(lambda x: x.sum(), 1)
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

    filledData.loc[:, 'blood'] = filledData.loc[:, days].apply(lambda x: x.sum(), 1)
    filledData.drop(columns=days, inplace=True)

    filledData.loc[:, "sex":"albumin"] = filledData.loc[:, "sex":"albumin"].fillna(filledData.loc[:, "sex":"albumin"].mean())
    filledData.fillna(0, inplace=True)
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

    filledData.loc[:, 'blood'] = filledData.loc[:, days].apply(lambda x: x.sum(), 1)
    filledData.drop(columns=days, inplace=True)

    filledData.loc[:, "sex":"albumin"] = filledData.loc[:, "sex":"albumin"].fillna(filledData.loc[:, "sex":"albumin"].median())
    filledData.fillna(0, inplace=True)
    filledData = filledData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
         'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites', 
         'chest', 'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']]
    
    return filledData
    
    
def DataFileWithKNN(rawdata, n_neighbors=2):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    print('n_neighbors', n_neighbors)

    days = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
            'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15']
    
    filledData = rawdata.copy().drop('blood_type', axis=1)

    filledData.loc[:, 'blood'] = filledData.loc[:, days].apply(lambda x: x.sum(), 1)
    filledData.drop(columns=days, inplace=True)

    # filledData.fillna(filledData.loc[:, :"lowerbody"].mean(), inplace=True)
    filledData = filledData[
        ['hospital', 'sex', 'age', 'weight', 'heart_rate', 'low_pressure',
         'high_pressure', 'temperature', 'hemoglobin', 'redcell', 'albumin',
         'upperbody', 'lowerbody', 'penetrate', 'pleural', 'ascites',
         'chest', 'abdomen', 'pelvis', 'c1', 'c2', 'c3', 'blood']]

    data_filled = imputer.fit_transform(filledData.to_numpy()[:, 1:])
    filledData.iloc[:, 1:] = data_filled

    return filledData
