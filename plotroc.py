import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

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
import pickle

# import networkx as nx
import random


def main(args, model_path_list=[], model_name_list=[],):

    pl.utilities.seed.seed_everything(1)
    info = [str(s) for s in sys.argv[1:]]
    runname = '111'

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

    train_data = dataset.data.cpu().numpy()
    val_data = dataset.dataval.cpu().numpy()
    test_data = dataset.datatest.cpu().numpy()
    train_label = dataset.label.cpu().numpy().astype(np.int32)
    val_label = dataset.labelval.cpu().numpy().astype(np.int32)
    test_label = dataset.labeltest.cpu().numpy().astype(np.int32)

    ns_probs = [0 for _ in range(len(test_label))]
    ns_auc = roc_auc_score(test_label, ns_probs)

    plt.figure(figsize=(5, 5))
    
    for i, modelpath in enumerate(model_path_list):
        classifier = pickle.load(open(modelpath, 'rb'))
        lr_probs = classifier.predict_proba(test_data)[:, 1]
        lr_auc = roc_auc_score(test_label, lr_probs)
        label_name = '%s: ROC AUC=%.3f' % (model_name_list[i], lr_auc)
        lr_fpr, lr_tpr, _ = roc_curve(test_label, lr_probs)
        plt.plot(lr_fpr, lr_tpr, marker='.', label=label_name)

    # calculate scores
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    ns_fpr, ns_tpr, _ = roc_curve(test_label, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()


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
    # parser.add_argument('--n_point', type=int, default=60000000, )
    # model param
    # parser.add_argument('--same_sigma', type=bool, default=False)
    # parser.add_argument('--show_detail', type=bool, default=False)
    # parser.add_argument('--plotInput', type=int, default=0)
    # parser.add_argument('--eta', type=float, default=0)
    # parser.add_argument('--NetworkStructure', type=list, default=[-1, 5000, 4000, 3000, 2000, 1000, 2])
    # parser.add_argument('--pow_input', type=float, default=2)
    # parser.add_argument('--pow_latent', type=float, default=2)
    # parser.add_argument('--near_bound', type=float, default=0.0)
    # parser.add_argument('--far_bound', type=float, default=1.0)

    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--method', type=str, default="dmt", choices=[
        'dmt',
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
    parser.add_argument('--p1', type=int, default=0)
    parser.add_argument('--p2', type=int, default=0)
    # parser.add_argument('--method', type=str, default='dmt',
    #                     choices=['dmt', 'dmt_mask'])
    parser.add_argument('--foldindex', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--fill_set', type=str, default='nNull', choices=[
        'nNull', 'fillWithMean', 'fillWithMiddle', 'fillWithKNN',
    ])

    parser.add_argument('--scale', type=float, default=30)
    parser.add_argument('--vs', type=float, default=1e-2)
    parser.add_argument('--ve', type=float, default=-1)
    parser.add_argument('--K', type=int, default=15)
    parser.add_argument("--uselabel", type=int, default=1)
    parser.add_argument("--classfication_model", type=int, default=1)

    # train param
    parser.add_argument('--batch_size', type=int, default=5000, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    # print(args.foldindex)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" if args.foldindex < 5 else "1"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" if args.foldindex < 5 else "1"

    df = pd.read_csv('file_path.csv')
    name_list = df['Unnamed: 0'].to_list()

    # path_list = df['nNull'].to_list()
    # main(args, model_path_list=path_list, model_name_list=name_list)
    # plt.savefig('roc/nNull.png')
    # plt.close()

    # df = pd.read_csv('file_path.csv')
    # args.__dict__['fill_set'] = 'fillWithMean'
    # path_list = df['fillWithMiddle'].to_list()
    # main(args, model_path_list=path_list, model_name_list=name_list)
    # plt.savefig('roc/fillWithMiddle.png')
    # plt.close()

    # df = pd.read_csv('file_path.csv')
    # args.__dict__['fill_set'] = 'fillWithMiddle'
    # path_list = df['fillWithKNN'].to_list()
    # main(args, model_path_list=path_list, model_name_list=name_list)
    # plt.savefig('roc/fillWithKNN.png')
    # plt.close()

    # df = pd.read_csv('file_path.csv')
    args.__dict__['fill_set'] = 'fillWithMean'
    path_list = df['fillWithMean'].to_list()
    main(args, model_path_list=path_list, model_name_list=name_list)
    plt.savefig('roc/fillWithMean.png')
    plt.close()
