
#%%%%

from cgi import test
from unicodedata import name
import pandas as pd 
import numpy as np
import wandb
import json
import plotly.graph_objects as go
import uuid
import matplotlib.pyplot as plt 
import plotly.express as px
import itertools
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os



def LoadInfo(paramName, metric_list, project_name='DAFS_v2', sweep='pxcw9b3s', plot_fig=''):
    api = wandb.Api()
    runs = api.sweep('cairi/' + project_name + '/' + sweep).runs
    summary_list, config_list, name_list = [], [], []

    for i, run in enumerate(runs):
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith('_')})

        print('{} of total {} runs'.format(i, len(runs)))

        # ima_name = ''
        if run.state == 'finished' or run.state == 'failed':
            # fig_inf = run.summary['main_easy/fig_easy']
            last_fig_name = 'joblib'
            for file in run.files():
                if last_fig_name in file.name:
                    image_path = 'Analysis/file/' + file.name
                    print(image_path)
                    config_list[-1]['image_path'] = image_path
                    if not os.path.exists(image_path):
                        file.download('Analysis/file/', replace=True)

    summary = pd.DataFrame(summary_list)
    config = pd.DataFrame(config_list)
    data = pd.concat([summary, config], axis=1)
    # import pdb; pdb.set_trace()
    return data[paramName + metric_list + ['image_path']]


def Any(wandb_data, train_str, test_str, mod='max'):
    
    print('wandb_data.shape', wandb_data.shape)
    mode = 'dis_pow'

    t_list = list(set(wandb_data[mode]))
    # t_list.remove(0.05)
    t_list.sort()
    data_name_list = list(set(wandb_data['data_name']))
    # data_name_list.remove('arcene')
    # data_name_list.remove('EMnistBC')

    data = np.zeros((len(data_name_list), len(t_list)))
    for i, data_name in enumerate(data_name_list):
        for j, ber in enumerate(t_list):
            try:
                wandb_data_0 = wandb_data[wandb_data[mode] == ber]
                data_select = wandb_data_0[wandb_data_0['data_name'] == data_name]
                if mod == 'max':
                    val_max = np.array(data_select[train_str].max())
                    test_best = data_select[data_select[train_str] == val_max][test_str].max()
                else:
                    val_max = np.array(data_select[train_str].min())
                    test_best = data_select[data_select[train_str] == val_max][test_str].min()
                data[i, j] = test_best
            except:
                data[i, j] = float(0)
    
    data_show = pd.DataFrame(data, index=data_name_list, columns=t_list).T
    data_show['Average'] = data_show.mean(axis=1)
    data_show = data_show.T
    print(data_show)
    return data_show

def G_table(wandb_data, item='best_test_auc'):

    fill_list = list(set(wandb_data['fill_set']))

    tab_dict = {}
    for f in fill_list:
        # f = 'fillWithKNN'
        wandb_data_fill = wandb_data[wandb_data['fill_set'] == f]
        method_list = list(set(wandb_data_fill['method']))
        
        tab_dict[f] = {}
        for m in method_list:
            # m = 'KNN'
            wandb_data_fill_model = wandb_data_fill[wandb_data_fill['method'] == m]
            best_auc_index = wandb_data_fill_model['best_test_auc'].argmax()
            best_auc = wandb_data_fill_model.iloc[best_auc_index][item]
            tab_dict[f][m] = best_auc

    return tab_dict

if __name__ == "__main__":

    # json2plotly()

    paramName=[
        'fill_set',
        'method',
        'p1',
        'p2',
    ]
    metric_list=[
        'best_test_auc',
        'best_test_acc',
    ]

    sweep_list = {
        'test_all_feature_v3': 'u9bqzvya', #'jyr2r525'
    }

    #%%
    test_str_list = [
        'train_cte',
    ]

    train_str = 'train_cte'
    # test_str = 'SVC_train'
    for test_str in test_str_list:
        data_show_list = []
        for sweep in sweep_list.keys():
            # print(sweep)
            wandb_data = LoadInfo(
                paramName,
                metric_list,
                project_name='bloodcenter_zzl',
                sweep=sweep_list[sweep],
                )
            
            tab_dict = G_table(wandb_data, item='best_test_auc')
            df = pd.DataFrame.from_dict(tab_dict)
            print(df.T)
            df.to_csv('best_test_auc.csv')

            tab_dict = G_table(wandb_data, item='best_test_acc')
            df = pd.DataFrame.from_dict(tab_dict)
            print(df.T)
            df.to_csv('best_test_acc.csv')

            tab_dict = G_table(wandb_data, item='image_path')
            df = pd.DataFrame.from_dict(tab_dict)
            print(df.T)
            df.to_csv('file_path.csv')
