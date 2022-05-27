import os
from time import sleep

import numpy as np
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import load_data_f.dataset as datasetfunc
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import sys

from datetime import date
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn import tree, svm, neighbors, linear_model, metrics
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier, XGBRegressor, DMatrix, plot_importance
import lightgbm as lgb
import time

# import networkx as nx
import random


def main(args):

    pl.utilities.seed.seed_everything(args.__dict__['seed'])
    info = [str(s) for s in sys.argv[1:]]
    runname = '_'.join(['dmt', args.data_name, ''.join(info)])

    disfunc_use = getattr(disfunc, 'EuclideanDistanceNumpy')
    simfunc_use = getattr(simfunc, 'UMAPSimilarity')
    simfunc_npuse = getattr(simfunc, 'UMAPSimilarityNumpy')
    # dm_class = getattr(datasetfunc, args.__dict__['data_name'] + 'DataModule')
    dm_class = getattr(datasetfunc, 'BlodNoMissingDataModule')

    dataset = dm_class(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        **args.__dict__,
    )

    if args.method == 'KNN':
        p1_list = [1, 3, 5, 10, 15, 20]
        p2_list = [10, 20, 30, 50, 70, 100]
    if args.method == 'lightGBM':
        p1_list = [21, 26, 31, 26, 41]
        p2_list = [80, 90, 100, 110, 120, 130]
    if args.method == 'LR':
        p1_list = [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.0007]
        p2_list = [0.6, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0]
    if args.method in ['random_forest', 'gradient_boost']:
        p1_list = [2, 3, 4, 5, 6, 7]
        p2_list = [80, 90, 100, 110, 120, 130]
    if args.method in ['decision_tree', 'extra_tree']:
        p1_list = [2, 3, 5, 7, 10, 15]
        p2_list = [1, 2, 3, 5, 7, 10]
    if args.method == 'adaboost':
        p1_list = [0.8, 0.9, 1, 2, 5, ]
        p2_list = [40, 50, 60, 70, 80, 90]
    if args.method == 'svm':
        p1_list = [10, 50, 100, 300, 500, -1]
        p2_list = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005]

    p1 = p1_list[args.p1]
    p2 = p2_list[args.p2]
    # p3 = p3_list[args.p3]

    refs = {
        'KNN': (neighbors.KNeighborsRegressor(n_neighbors=p1, leaf_size=p2), neighbors.KNeighborsClassifier(n_neighbors=p1, leaf_size=p2)),
        'LR': (linear_model.LinearRegression(), linear_model.LogisticRegression(tol=p1, C=p2, penalty='l2', random_state=1)),
        'random_forest': (RandomForestRegressor(min_samples_split=p1, n_estimators=p2, random_state=1), RandomForestClassifier(max_depth=p1, n_estimators=p2, random_state=1)),
        'decision_tree': (tree.DecisionTreeRegressor(min_samples_split=p1, min_samples_leaf=p2, random_state=1), tree.DecisionTreeClassifier(min_samples_split=p1, min_samples_leaf=p2, random_state=1)),
        'extra_tree': (tree.ExtraTreeRegressor(min_samples_split=p1, min_samples_leaf=p2, random_state=1), tree.ExtraTreeClassifier(min_samples_split=p1, min_samples_leaf=p2, random_state=1)),
        'svm': (svm.SVR(), svm.SVC(max_iter=p1, tol=p2, random_state=1, probability=True)),
        'gradient_boost': (GradientBoostingRegressor(min_samples_split=p1, n_estimators=p2, learning_rate=1.0, max_depth=1, random_state=1), GradientBoostingClassifier(min_samples_split=p1, n_estimators=p2, learning_rate=1.0, max_depth=1, random_state=1)),
        'adaboost': (AdaBoostRegressor(learning_rate=p1, n_estimators=p2, random_state=1), AdaBoostClassifier(learning_rate=p1, n_estimators=p2, random_state=1)),
        'lightGBM': (lgb.LGBMRegressor(num_leaves=p1, n_estimators=p2, random_state=1), lgb.LGBMClassifier(num_leaves=p1, n_estimators=p2, random_state=1)),
    }

    train_data = dataset.data.cpu().numpy()
    val_data = dataset.dataval.cpu().numpy()
    test_data = dataset.datatest.cpu().numpy()
    train_label = dataset.label.cpu().numpy().astype(np.int32)
    val_label = dataset.labelval.cpu().numpy().astype(np.int32)
    test_label = dataset.labeltest.cpu().numpy().astype(np.int32)

    clf = refs[args.method][1]
    clf.fit(train_data, train_label)

    train_predict = clf.predict_proba(train_data)[:, 1]
    train_fpr, train_tpr, thresholds = metrics.roc_curve(train_label, train_predict)
    train_predict[train_predict < 0.5] = 0
    train_predict[train_predict > 0.5] = 1
    train_score = metrics.accuracy_score(train_label, train_predict.astype(np.int))
    train_auc = metrics.auc(train_fpr, train_tpr)

    val_predict = clf.predict_proba(val_data)[:, 1]
    val_fpr, val_tpr, thresholds = metrics.roc_curve(val_label, val_predict)
    val_predict[val_predict < 0.5] = 0
    val_predict[val_predict > 0.5] = 1
    val_score = metrics.accuracy_score(val_label, val_predict.astype(np.int))
    val_auc = metrics.auc(val_fpr, val_tpr)

    test_predict = clf.predict_proba(test_data)[:, 1]
    test_fpr, test_tpr, thresholds = metrics.roc_curve(test_label, test_predict,)
    test_predict[test_predict < 0.5] = 0
    test_predict[test_predict > 0.5] = 1
    test_score = metrics.accuracy_score(test_label, test_predict.astype(np.int))
    test_auc = metrics.auc(test_fpr, test_tpr)

    log_dict = {
        'train_acc': train_score,
        'train_auc': train_auc,
        'val_acc': val_score,
        'val_auc': val_auc,
        'test_acc': test_score,
        'test_auc': test_auc,
    }
    return log_dict, clf
    # wandb.log()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='digits_T', )

    # data set param
    parser.add_argument('--data_name', type=str, default='BlodNNull',
                        # choices=[
                        #     # 'Digits', 'Coil20', 'Coil100',
                        #     # 'Smile', 'ToyDiff', 'SwissRoll',
                        #     # 'KMnist', 'EMnist', 'Mnist',
                        #     # 'EMnistBC', 'EMnistBYCLASS',
                        #     # 'Cifar10', 'Colon',
                        #     # 'Gast10k', 'HCL60K', 'PBMC',
                        #     # 'HCL280K', 'SAMUSIK',
                        #     # 'M_handwritten', 'Seversphere',
                        #     # 'MCA', 'Activity', 'SwissRoll2',
                        #     # 'PeiHuman', 'BlodZHEER', 'BlodAll'
                        #     'BlodNoMissing',]
                        )

    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--method', type=str, default="KNN", choices=[
        'KNN',
        'LR',
        'random_forest',
        'decision_tree',
        'extra_tree',
        'svm',
        'gradient_boost',
        'adaboost',
        'lightGBM',
        'xgboost',
        'bagging',
    ])
    parser.add_argument('--v_input', type=float, default=100)
    parser.add_argument('--perplexity', type=int, default=20)
    parser.add_argument('--NetworkStructure_1', type=list, default=[-1, 500, 300, 80])
    parser.add_argument('--NetworkStructure_2', type=list, default=[-1, 500, 80])
    parser.add_argument('--num_latent_dim', type=int, default=2)
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--augNearRate', type=float, default=100)
    parser.add_argument('--offline', type=int, default=0)
    # parser.add_argument('--method', type=str, default='dmt',
    #                     choices=['dmt', 'dmt_mask'])
    parser.add_argument('--foldindex', type=int, default=0)
    parser.add_argument('--p1', type=int, default=0)
    parser.add_argument('--p2', type=int, default=0)
    parser.add_argument('--hospitals', type=str, default="all", choices=[
        'all',
        'DYang',
        'EZe',
        'HNing',
        'SYi',
        'SYiFu',
        'SYu',
        'WLing',
        'XChang',
        'XShan',
        'YKang',
        'YYao',
        'ZEr'
    ])

    # parser.add_argument('--scale', type=int, default=30)
    # parser.add_argument('--vs', type=float, default=1e-2)
    # parser.add_argument('--ve', type=float, default=-1)
    # parser.add_argument('--K', type=int, default=15)

    # train param
    # parser.add_argument('--batch_size', type=int, default=2000, )
    # parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    # parser.add_argument('--seed', type=int, default=1, metavar='S')
    # parser.add_argument('--log_interval', type=int, default=10)
    # parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    print(args.foldindex)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" if args.foldindex < 5 else "1"
    main(args)
