#! /usr/bin/env python
from datetime import date
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score
from sklearn import tree, svm, neighbors, linear_model, metrics
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier, XGBRegressor, DMatrix, plot_importance
import lightgbm as lgb
import time

from loaddata import DataFileWithKNN, preprocess
from copy import deepcopy
import matplotlib.pyplot as plt


def drawROC(fprs, tprs):
    plt.figure(figsize=(10.2, 7.6))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    for clf_key in refs.keys():
        plt.plot(fprs[clf_key], tprs[clf_key], lw=2, label=clf_key)
    plt.legend()
    plt.show()


def traditionWay(refs: dict, data, label):
    def try_different_method_clf(clf, train_data: np.ndarray, train_label: np.ndarray, val_data: np.ndarray, val_label: np.ndarray, test_data: np.ndarray, test_label: np.ndarray, cnt):
        # print(type(train_label))
        if clf is None:
            return 0, 0, 0, 0, 0, 0, 0, 0

        train_label_local, val_label_local, test_label_local = deepcopy(train_label), deepcopy(val_label), deepcopy(test_label)
        train_label_local[train_label_local > 0] = 1
        val_label_local[val_label_local > 0] = 1
        test_label_local[test_label_local > 0] = 1

        clf.fit(train_data, train_label_local)
        val_score = clf.score(val_data, val_label_local)
        test_score = clf.score(test_data, test_label_local)
        val_predict_label = clf.predict(val_data)
        test_predict_label = clf.predict(test_data)

        val_fpr, val_tpr, thresholds = metrics.roc_curve(val_predict_label, val_label_local)
        val_auc = metrics.auc(val_fpr, val_tpr)
        fontsize = 14
        ax1.plot(val_fpr, val_tpr, lw=2, label="{}_{}_sc: {:5.2f}".format(k, i, val_score))
        ax1.set_xlabel('False Positive Rate', fontsize=fontsize)
        ax1.set_ylabel('True Positive Rate', fontsize=fontsize)
        ax1.set_title(f'ROC Curve of {k} with {tl} Data')
        ax1.legend()

        val_precision, val_recall, thresholds = precision_recall_curve(val_label_local, val_predict_label)
        ax2.plot(val_recall, val_precision, lw=2, label=k + ' (area = %0.2f)' % average_precision_score(val_label_local, val_predict_label))
        ax2.set_xlabel('Recall', fontsize=fontsize)
        ax2.set_ylabel('Precision', fontsize=fontsize)
        ax2.set_title(f'Precision Recall Curve of {k} with {tl} Data')
        ax2.legend()
        # print(precision, recall)

        test_fpr, test_tpr, thresholds = metrics.roc_curve(test_predict_label, test_label_local)
        test_auc = metrics.auc(test_fpr, test_tpr)
        test_precision, test_recall, thresholds = precision_recall_curve(test_label_local, test_predict_label)
        return val_score, test_score, val_auc, test_auc, test_fpr, test_tpr, test_precision, test_recall

    def try_different_method_refs(refs, train_data: np.ndarray, train_label: np.ndarray, val_data: np.ndarray, val_label: np.ndarray, test_data: np.ndarray, test_label: np.ndarray):
        refs.fit(train_data, train_label)
        pre_result = refs.predict(test_data)
        pre_result[pre_result < 0] = 0
        score = mean_squared_error(pre_result, test_label)
        return score

    skf = StratifiedKFold(n_splits=10)
    local_label = deepcopy(label)
    local_label[local_label > 0] = 1
    label[label > 0] = 1
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, stratify=local_label, random_state=1)
    # print(X_train, X_test, y_train, y_test)

    rows = len([i[1] for i in refs.values() if i[1] is not None])
    # rows = len(refs.values())
    # plt.figure(figsize=(20.4, 3.8 * rows))
    plt.style.use('seaborn')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    fig1 = plt.figure(figsize=(10.2, 3.8 * rows))
    fig2 = plt.figure(figsize=(10.2, 3.8 * rows))
    # plt.figure()
    cnt = 1
    for k, v in refs.items():
        acc_cl, test_acc_cl, auc_cl, test_auc_cl, test_fprs, test_tprs, test_pres, test_recs, acc_re = [], [], [], [], [], [], [], [], []
        # print('Classifier|Regressor: {:15s}'.format(key))

        # for train_index, test_index in kf.split(data):
        # train_X, test_X = data[train_index], data[test_index]
        # for train_index, test_index in kf.split(data):
        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.05)
        if v[1] is not None:
            ax1 = fig1.add_subplot(rows, 2, cnt)
            ax2 = fig2.add_subplot(rows, 2, cnt)
        for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
            if v[1] is not None:
                ac_cl, test_ac_cl, au_cl, test_au_cl, test_fpr, test_tpr, test_pre, test_rec = try_different_method_clf(v[1], X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index], X_test, y_test, cnt)

                acc_cl.append(ac_cl)
                test_acc_cl.append(test_ac_cl)
                auc_cl.append(au_cl)
                test_auc_cl.append(test_au_cl)
                test_fprs.append(test_fpr)
                test_tprs.append(test_tpr)
                test_pres.append(test_pre)
                test_recs.append(test_rec)

            # if v[0] is not None:
            #     ac_re = try_different_method_refs(v[0], X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index])
            #     acc_re.append(ac_re)

            #     ax = plt.subplot(rows, 3, cnt + 2)
            #     ax.plot(fpr, tpr, lw=2, label=k + "_" + str(i) + "_score: " + str(ac_cl))
            #     ax.set_xlabel('False Positive Rate')
            #     ax.set_ylabel('True Positive Rate')
            #     ax.set_title(k)
            #     ax.legend()

            #     acc_cl.append(ac_cl)
            #     acc_test_cl.append(ac_test_cl)
            #     auc_cl.append(au_cl)
            #     auc_test_cl.append(au_test_cl)
        # result[k] = (acc_cl, auc_cl, acc_re)
        if v[1] is not None:
            idx = acc_cl.index(np.max(acc_cl))
            print(f'clf:{k}, acc_cl: {acc_cl}, best model: {idx}')
            ax3 = fig1.add_subplot(rows, 2, cnt + 1)
            ax3.plot(test_fprs[idx], test_tprs[idx], lw=2, label="{}_{}_SC: {:5.2f}_AUC: {:5.2f}".format(k, idx, test_acc_cl[idx], test_auc_cl[idx]))
            ax3.set_xlabel('False Positive Rate', fontsize=fontsize)
            ax3.set_ylabel('True Positive Rate', fontsize=fontsize)
            ax3.set_title(f'ROC result of {k} with Test {tl} Data')
            ax3.legend()
            ax4 = fig2.add_subplot(rows, 2, cnt + 1)
            ax4.plot(test_pres[idx], test_recs[idx], lw=2, label='{}_{}'.format(k, idx))
            ax4.set_xlabel('False Positive Rate', fontsize=fontsize)
            ax4.set_ylabel('True Positive Rate', fontsize=fontsize)
            ax4.set_title(f'Precision Recall Curve of {k} with Test {tl} Data')
            ax4.legend()
            cnt += 2
            f.write("Classifior: {:15s}, best model: {:6.3f}, acc mean: {:6.3f}, acc_val: {:6.3f}, acc_test: {:6.3f}, auc_test: {:6.3f}, max: {:6.3f}, min: {:6.3f}, auc mean: {:6.3f}\n".format(k, idx, np.mean(acc_cl), np.std(acc_cl), test_acc_cl[idx], test_auc_cl[idx], np.max(acc_cl), np.min(acc_cl), np.mean(auc_cl)))

        # if v[0] is not None:
        #     f.write("Regressor : {:15s}, mse_mean: {:6.3f}, acc_val: {:6.3f}, max: {:6.3f}, min: {:6.3f}\n".format(k, np.mean(acc_re), np.std(acc_re), np.max(acc_re), np.min(acc_re)))
    # plt.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f"log/pic/ROC_{'_'.join(tl.split())}.jpg")
    fig2.savefig(f"log/pic/PR_{'_'.join(tl.split())}.jpg")
    # plt.show()

    # print(result)
    # for key, (acc_cl, auc_cl, acc_re) in result.items():
    #     if refs[key][1] is not None:
    #         f.write("Classifior: {:15s}, acc mean: {:6.3f}, acc_val: {:6.3f}, max: {:6.3f}, min: {:6.3f}, auc: {:6.3f}\n".format(
    #             key, np.mean(acc_cl), np.std(acc_cl), np.max(acc_cl), np.min(acc_cl), np.mean(auc_cl)))
    #         print("Classifior: {:15s}, acc mean: {:6.3f}, acc_val: {:6.3f}, max: {:6.3f}, min: {:6.3f}, auc: {:6.3f}\n".format(
    #             key, np.mean(acc_cl), np.std(acc_cl), np.max(acc_cl), np.min(acc_cl), np.mean(auc_cl)))

    # for key, (acc_cl, auc_cl, acc_re) in result.items():
    #     f.write("Regressor : {:15s}, mse_mean: {:6.3f}, acc_val: {:6.3f}, max: {:6.3f}, min: {:6.3f}\n".format(
    #         key, np.mean(acc_re), np.std(acc_re), np.max(acc_re), np.min(acc_re)))
    #     print("Regressor : {:15s}, mse_mean: {:6.3f}, acc_val: {:6.3f}, max: {:6.3f}, min: {:6.3f}\n".format(
    #         key, np.mean(acc_re), np.std(acc_re), np.max(acc_re), np.min(acc_re)))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('-n', '--knn_neighbors', type=int, default=2, )
    parser.add_argument('-i', '--if_norm', type=bool, default=True, )
    # args = argparse.add_argparse_args(parser)
    args = parser.parse_args()
    fontsize = 14

    # (
    #     nNullData, nNullLabel,
    #     fillWithMeanData, fillWithMeanLabel,
    #     fillWithMiddleData, fillWithMiddleLabel,
    #     fillWithKNNData, fillWithKNNLabel,
    #  ) = preprocess(knn_neighbors=args.knn_neighbors, ifNorm=args.if_norm)

    (
        nNull, missing, fillWithMean, fillWithMiddle, fillWithKNN,
    ) = preprocess(knn_neighbors=args.knn_neighbors, ifNorm=args.if_norm, preSave=True)

    # data, label = nNull['data'], nNull['label']
    # tl = "Not Null"
    data, label = fillWithMean['data'], fillWithMean['label']
    tl = "Fill With Mean"
    # data, label = fillWithMiddle['data'], fillWithMiddle['label']
    # tl = "Fill With Middle"

    refs = {'KNN': (neighbors.KNeighborsRegressor(n_neighbors=3), neighbors.KNeighborsClassifier(n_neighbors=3)),
            'LR': (linear_model.LinearRegression(), linear_model.LogisticRegression(penalty='l2', random_state=1)),
            'random_forest': (RandomForestRegressor(n_estimators=50, random_state=1), RandomForestClassifier(n_estimators=50, random_state=1)),
            'decision_tree': (tree.DecisionTreeRegressor(random_state=1), tree.DecisionTreeClassifier(random_state=1)),
            'extra_tree': (tree.ExtraTreeRegressor(random_state=1), tree.ExtraTreeClassifier(random_state=1)),
            'svm': (svm.SVR(), svm.SVC(random_state=1)),
            'gradient_boost': (GradientBoostingRegressor(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=1), GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=1)),
            'adaboost': (AdaBoostRegressor(n_estimators=50, random_state=1), AdaBoostClassifier(n_estimators=50, random_state=1)),
            'lightGBM': (lgb.LGBMRegressor(random_state=1), lgb.LGBMClassifier(random_state=1)),
            'xgboost': (XGBRegressor(verbosity=0), XGBClassifier(verbosity=0, use_label_encoder=False)),
            'bagging': (BaggingRegressor(verbose=0, random_state=1), BaggingClassifier(verbose=0, random_state=1))
            }

#         'naive_gaussian': naive_bayes.GaussianNB(), \
#         'naive_mul':naive_bayes.MultinomialNB(),\
#         'bagging_knn' : BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5), \
#         'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),

    fileName = time.strftime('log/%y%b%d_%I%M%p.log', time.localtime())
    try:
        with open(fileName, 'w') as f:
            f.write(f"Processing {tl} data\n")
            # traditionWay(refs, datatrain, labeltrain, datatest, labeltest)
            traditionWay(refs, data, label)
            print('----------------------------')
            # f.write("\nProcessing filled data with mean values\n")
            # traditionWay(refs, fillWithMeanData, fillWithMeanLabel)
            # print('----------------------------')
            # f.write("\nProcessing filled data with middle values\n")
            # traditionWay(refs, fillWithMiddleData, fillWithMiddleLabel)
            # f.write("\n\nProcessing Fill With Mean data\n")
            # traditionWay_cl(clfs, nNullData, nNullLabel)
            # f.write("\nProcessing filled data with mean values\n")
            # traditionWay_cl(clfs, fillWithMeanData, fillWithMeanLabel)
            # f.write("\nProcessing filled data with middle values\n")
            # traditionWay_cl(clfs, fillWithMiddleData, fillWithMiddleLabel)
            # f.write("\nProcessing filled data with knn value\n")
            # traditionWay(clfs, fillWithKNNData, fillWithKNNLabel)
    except Exception as e:
        print(repr(e))
        import os
        os.remove(fileName)
