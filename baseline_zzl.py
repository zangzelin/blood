from datetime import datetime
import os
import shutil
from time import sleep

import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
import numpy as np
import functools
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
import Loss.dmt_loss_aug as dmt_loss_aug

# import Similarity.sim_tdis as Sim_use
import Similarity.sim_Gaussian_1 as Sim_use
import Loss.dmt_loss_old as Loss_use
import Loss.dmt_loss_nearfar as Loss_use_mask
import nuscheduler
import wandb
import eval.eval_core as eval_core
import plotly.express as px
import matplotlib.pylab as plt
import pandas as pd

import load_data_f.dataset as datasetfunc
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import sys
import git
from sklearn import metrics

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

# import networkx as nx
import random
from pytorch_lightning.callbacks import EarlyStopping

torch.set_num_threads(1)


class DMT_Model(pl.LightningModule):

    def __init__(
        self,
        dataset,
        DistanceF,
        SimilarityF,
        data_dir='./data',
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.celoss = nn.CrossEntropyLoss()
        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = self.hparams.lr
        self.train_batchsize = self.hparams.batch_size
        self.test_batchsize = self.hparams.batch_size
        # self.save_path = save_path
        # self.num_latent_dim = 2
        self.dims = dataset.data[0].shape
        self.log_interval = self.hparams.log_interval

        self.train_dataset = dataset
        self.DistanceF = DistanceF
        self.SimilarityF = SimilarityF
        # self.hparams.NetworkStructure.append(
        #     self.hparams.num_latent_dim
        #     )
        self.model, self.model2, self.model3 = self.InitNetworkMLP(self.hparams.NetworkStructure_1, self.hparams.NetworkStructure_2)
        # print(self.hparams.NetworkStructure)
        print(self.model)
        print(self.model2)
        print(self.model3)
        # if self.hparams.method == 'dmt':
        self.Loss = dmt_loss_aug.MyLoss(
            v_input=self.hparams.v_input,
            SimilarityFunc=SimilarityF,
            metric=self.hparams.metric,
            augNearRate=self.hparams.augNearRate,
            # eta=self.hparams.eta,
        )
        self.wandb_logs = {}
        self.val_wandb_logs = {}
        self.path_list = None
        # if self.hparams.method == 'dmt_mask':
        # self.Loss = Loss_use_mask.MyLoss(
        #     v_input=self.hparams.v_input,
        #     SimilarityFunc=SimilarityF,
        #     metric=self.hparams.metric,
        #     near_bound=self.hparams.near_bound,
        #     far_bound=self.hparams.far_bound,
        #     )

        self.nushadular = nuscheduler.Nushadular(
            nu_start=self.hparams.vs,
            nu_end=self.hparams.ve,
            epoch_start=self.hparams.epochs // 5,
            epoch_end=self.hparams.epochs * 4 // 5,
        )
        self.l_shadular = nuscheduler.Nushadular(
            nu_start=0.000001,
            nu_end=100,
            epoch_start=self.hparams.epochs // 5,
            epoch_end=self.hparams.epochs * 4 // 5,
        )

        self.labelstr = dataset.label_str

    # def Curance_path_list(self, neighbour_input, distance_input):

    #     row = []
    #     col = []
    #     v = []
    #     n_p, n_n = neighbour_input.shape
    #     for i in range(n_p):
    #         for j in range(n_n):
    #             row.append(i)
    #             col.append(neighbour_input[i,j])
    #             v.append(distance_input[i,j])

    #     G=nx.Graph()
    #     for i in range(0, n_p):
    #         G.add_node(i)
    #     for i in range(len(row)):
    #         G.add_weighted_edges_from([(row[i],col[i],v[i])])

    #     # pos=nx.shell_layout(G)
    #     # nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5 )

    #     path_list = []
    #     for i in range(5000):
    #         source = random.randint(a=0, b=n_p-1)
    #         target = random.randint(a=0, b=n_p-1)
    #         try:
    #             path=nx.dijkstra_path(G, source=source, target=target)
    #             path_list.append(path)
    #         except:
    #             pass

    #     return path_list

    def forward(self, x):
        lat1 = x
        for i, m in enumerate(self.model):
            lat1 = m(lat1)
            # if i == 0:
            #     mid1 = lat1
            # else:
            #     mid1 += lat1

        lat2 = lat1
        for i, m in enumerate(self.model2):
            lat2 = m(lat2)
            # if i == 0:
            #     mid2 = lat2
            # else:
            #     mid2 += lat2

        lat3 = lat2
        for i, m in enumerate(self.model3):
            lat3 = m(lat3)

        return lat1, lat3

    def aug_near_mix(self, index, dataset, stage="train", k=10):

        r = (torch.arange(start=0, end=index.shape[0]) * k + torch.randint(low=1, high=k, size=(index.shape[0],))).cuda()
        if stage == "train":
            random_select_near_index = dataset.neighbors_index[index][:, :k].reshape((-1,))[r].long()
            random_select_near_data2 = dataset.data[random_select_near_index]
            random_rate = torch.rand(size=(index.shape[0], 1)).cuda()
            return random_rate * random_select_near_data2 + (1 - random_rate) * dataset.data[index]
        elif stage == "val":
            random_select_near_index = dataset.neighbors_index_val[index][:, :k].reshape((-1,))[r].long()
            random_select_near_data2 = dataset.labelval[random_select_near_index]
            random_rate = torch.rand(size=(index.shape[0], 1)).cuda()
            return random_rate * random_select_near_data2 + (1 - random_rate) * dataset.labelval[index]

    def training_step(self, batch, batch_idx):
        # data1, data2, rho, sigma, label, index = batch
        index = batch

        data1 = self.data_train.data[index]
        data2 = self.aug_near_mix(index, self.data_train, k=self.hparams.K)
        # label1 = torch.tensor(self.data_train.label)[index]
        # label2 = torch.tensor(self.data_train.label)[index]
        label1 = self.data_train.label[index].clone()
        label2 = self.data_train.label[index].clone()

        # rho = self.data_train.rho[index]
        # sigma = self.data_train.sigma[index]
        # label = np.array(self.data_train.label)[index.cpu().numpy()]
        # label = self.data_train.label[index]

        mid1, lat1 = self(data1)
        mid2, lat2 = self(data2)
        data = torch.cat([mid1, mid2]).to(self.device)
        lat = torch.cat([lat1, lat2]).to(self.device)
        label = torch.cat([label1, label2]).to(self.device)

        # print(lat)
        # latval = nn.Softmax()(lat)
        # latval = latent

        # predictint = latval[:, 1].clone()
        # predictint[predictint > 0.5] = 1
        # predictint[predictint <= 0.5] = 0

        loss = self.Loss(
            input_data=data.reshape(data.shape[0], -1),
            latent_data=lat.reshape(lat.shape[0], -1),
            rho=0.0,  # torch.cat([rho, rho]),
            sigma=1.0,  # torch.cat([sigma, sigma]),
            v_latent=self.nushadular.Getnu(self.current_epoch),
        )
        # print(predictint.cpu().long(), '\n', label.long())
        loss_ce = self.celoss(lat, label.long())
        total_loss = loss + loss_ce / self.hparams.scale

        # self.logger.experiment.
        self.wandb_logs = {
            'loss': loss,
            'loss_ce': loss_ce,
            'total_loss': total_loss,
            'nv': self.nushadular.Getnu(self.current_epoch),
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'dimention_loss': self.l_shadular.Getnu(self.current_epoch),
            'epoch': self.current_epoch,
            'visualize/train_embdeing': px.scatter(x=lat1[:, 0].cpu().detach().numpy(), y=lat1[:, 1].cpu().detach().numpy(), color=label1.cpu().detach().numpy())
            # 'visualize/train_valembdeing' : px.scatter(x=lattest[:, 0], y=lattest[:, 1], color=np.array(label))

        }
        # print(self.wandb_logs)
        wandb.log(self.wandb_logs)

        # if self.hparams.NetworkStructure[-1] >2:
        #     loss += torch.mean(lat1[:, 2:]) * self.l_shadular.Getnu(self.current_epoch)
        # self.scheduler.step()
        return total_loss

    def validation_step(self, batch, batch_idx):
        # print("batchsize: {}, shape:{}".format(batch.shape[0] , self.data_train.datatest.shape[0]))
        index = batch if batch.shape[0] < self.data_val.dataval.shape[0] else torch.Tensor(range(self.data_val.dataval.shape[0])).long()
        # print(batch, torch.Tensor(range(self.data_val.dataval.shape[0])).long())

        # print(index)
        data1 = self.data_val.dataval[index].to(self.device)
        # data2 = self.aug_near_mix(index, self.data_val, stage="val", k=self.hparams.K).to(self.device)
        # data2 = self.data_val.data[index]
        # rho = self.data_val.rho[index]
        # sigma = self.data_val.sigma[index]
        label1 = np.array(self.data_val.labelval)
        ind = index.cpu()
        ind = ind.numpy()
        label1 = label1[ind]
        # label2 = np.array(self.data_val.labelval)[index.cpu().numpy()]
        # latent = self.model(data)
        mid1, lat1 = self(data1)
        # mid2, lat2 = self(data2)

        # data = torch.cat([mid1, mid2]).to(self.device)
        # lat = torch.cat([lat1, lat2]).to(self.device)
        # label = torch.cat([label1, label2]).to(self.device)
        data = mid1.to(self.device)
        lat = lat1.to(self.device)
        label = torch.Tensor(label1).to(self.device)

        return (
            data,
            lat,
            # lat2.detach().cpu().numpy(),
            label,
            index,
        )

    def validation_epoch_end(self, outputs):

        # print('self..current_epoch',self.current_epoch)
        if (self.current_epoch + 1) % self.log_interval == 0:

            data = torch.cat([data_item[0] for data_item in outputs])
            latent = torch.cat([data_item[1] for data_item in outputs])
            # latent2 = np.concatenate([data_item[2] for data_item in outputs])
            label = torch.cat([data_item[2] for data_item in outputs])
            index = torch.cat([data_item[3] for data_item in outputs])

            e = eval_core.Eval(
                input=data,
                latent=latent,
                label=label,
                k=10)

            # dataval = self.data_val.dataval
            # midval, latval = self(dataval.to(self.device))
            # labelval = self.data_val.labelval.astype(np.int32)

            # latval = latent

            from sklearn import metrics
            # fpr, tpr, thresholds = metrics.roc_curve(
            #     latval[:,0], labelval
            #     )

            loss = self.Loss(
                input_data=data.reshape(data.shape[0], -1),
                latent_data=latent.reshape(data.shape[0], -1),
                rho=0.0,  # torch.cat([rho, rho]),
                sigma=1.0,  # torch.cat([sigma, sigma]),
                v_latent=self.nushadular.Getnu(self.current_epoch),
            )
            loss_ce = self.celoss(latent, label.long())
            total_loss = loss + loss_ce / self.hparams.scale

            lattest = nn.Softmax(dim=1)(latent).cpu().numpy()
            predictint = lattest[:, 1].copy()
            predictint[predictint > 0.5] = 1
            predictint[predictint <= 0.5] = 0

            label = label.cpu().numpy()
            latent = latent.cpu().numpy()

            try:
                self.wandb_logs.update({
                    'val_loss': loss,
                    'val_loss_ce': loss_ce,
                    'val_total_loss': total_loss,
                    # 'epoch': self.current_epoch,
                    # 'foldindex': self.test_dataset.foldindex,
                    'metric/val_auc': metrics.roc_auc_score(label, latent[:, 1]),
                    'metric/val_acc': metrics.accuracy_score(label, predictint)
                })
                # self.val_wandb_logs = {
                #     'val_loss': loss,
                #     'val_loss_ce': loss_ce,
                #     'val_total_loss': total_loss,
                #     'epoch': self.current_epoch,
                #     # 'foldindex': self.test_dataset.foldindex,
                #     'metric/val_auc': metrics.roc_auc_score(label, latent[:, 1]),
                #     'metric/val_acc': metrics.accuracy_score(label, predictint)
                # }
            except ValueError as e:
                print("error: ", repr(e))

            # e = eval_core.Eval(
            #     input=data,
            #     latent=latent,
            #     label=label,
            #     k=10
            #     )
            # self.wandb_logs['metric/AccSvc_dimall'] = e.E_Classifacation_SVC()

            for i in range(len(self.labelstr)):

                # if self.hparams.plotInput == 1:
                #     self.hparams.plotInput = 0
                #     if latent.shape[1] == 3:
                #         self.val_wandb_logs['visualize/val_Inputembdeing{}'.format(str(i))] = px.scatter_3d(
                #             x=data[:, 0], y=data[:, 1], z=data[:, 2],
                #             color=np.array(self.labelstr[i])[index],
                #             color_continuous_scale='speed'
                #         )

                if latent.shape[1] == 2:
                    self.wandb_logs['visualize/val_embdeing{}'.format(str(i))] = px.scatter(
                        x=latent[:, 0], y=latent[:, 1], color=np.array(self.labelstr[i])[index])
                    self.wandb_logs['visualize/val_valembdeing{}'.format(str(i))] = px.scatter(
                        x=lattest[:, 0], y=lattest[:, 1], color=np.array(label))

                elif latent.shape[1] == 3:
                    self.wandb_logs['visualize/val_embdeing{}'.format(str(i))] = px.scatter_3d(
                        x=latent[:, 0], y=latent[:, 1], z=latent[:, 2], color=np.array(self.labelstr[i])[index])

            wandb.log(self.wandb_logs)

    def test_step(self, batch, batch_idx):
        # print("batchsize: {}, shape:{}".format(batch.shape[0] , self.data_train.datatest.shape[0]))
        index = batch if batch.shape[0] < self.data_test.datatest.shape[0] else torch.Tensor(range(self.data_test.datatest.shape[0])).long().to(self.device)

        # print(index)
        data = self.data_test.datatest[index].to(self.device)
        # data2 = self.aug_near_mix(index, self.data_test, k=self.hparams.K).to(self.device)
        # data2 = self.data_test.data[index]
        # rho = self.data_test.rho[index]
        # sigma = self.data_test.sigma[index]
        label1 = np.array(self.data_test.labeltest)[index.cpu().numpy()]
        # latent = self.model(data)
        mid1, lat1 = self(data)
        # mid2, lat2 = self(data2)

        data = mid1.to(self.device)
        lat = lat1.to(self.device)
        label = torch.Tensor(label1).to(self.device)

        return (
            data,
            lat,
            # lat2.detach().cpu().numpy(),
            label,
            index,
        )

    def test_epoch_end(self, outputs):

        # print('self.current_epoch',self.current_epoch)
        if (self.current_epoch + 1) % self.log_interval == 0:

            # data = np.concatenate([data_item[0] for data_item in outputs])
            # latent = np.concatenate([data_item[1] for data_item in outputs])
            # # latent2 = np.concatenate([data_item[2] for data_item in outputs])
            # label = np.concatenate([data_item[2] for data_item in outputs])
            # index = np.concatenate([data_item[3] for data_item in outputs])

            data = torch.cat([data_item[0] for data_item in outputs])
            latent = torch.cat([data_item[1] for data_item in outputs])
            # latent2 = np.concatenate([data_item[2] for data_item in outputs])
            label = torch.cat([data_item[2] for data_item in outputs])
            index = torch.cat([data_item[3] for data_item in outputs])

            e = eval_core.Eval(
                input=data,
                latent=latent,
                label=label,
                k=10)

            # datatest = self.data_test.datatest
            # midtest, lattest = self(datatest.to(self.device))
            # labeltest = self.data_test.labeltest.astype(np.int32)

            # lattest = nn.Softmax()(torch.Tensor(latent))
            # lattest = lattest.cpu().numpy()

            # predictint = lattest[:, 1].clone()
            # predictint[predictint > 0.5] = 1
            # predictint[predictint <= 0.5] = 0
            # if self.path_list == None:
            # fpr, tpr, thresholds = metrics.roc_curve(
            #     lattest[:,0], labeltest
            #     )

            lattest = nn.Softmax(dim=1)(latent).cpu().numpy()
            predictint = lattest[:, 1].copy()
            predictint[predictint > 0.5] = 1
            predictint[predictint <= 0.5] = 0

            label = label.cpu().numpy()
            latent = latent.cpu().numpy()

            # now = os.path.join(f"{args.scale}_{args.vs}_{args.K}", datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + f"_{args.foldindex}_{args.data_name}")
            # if not os.path.exists(f"log/{now}/visualize"):
            #     os.makedirs(f"log/{now}/visualize")

            try:
                self.wandb_logs.update({
                    # 'epoch': self.current_epoch,
                    # 'foldindex': self.test_dataset.foldindex,
                    'metric/test_auc': metrics.roc_auc_score(label, latent[:, 1]),
                    'metric/test_acc': metrics.accuracy_score(label, predictint)
                    # 'metric/Mrremean': np.mean(e.E_mrre()),
                    # 'metric/Continuity': e.E_continuity(),
                    # 'metric/Trustworthiness':e.E_trustworthiness(),
                    # 'metric/Pearson': e.E_Rscore(),
                    # 'SVC':e.E_Classifacation_SVC(),
                    # 'Curance':e.E_Curance("SwissRoll" in self.hparams.data_name),
                    # 'Curance_2':e.E_Curance_2("SwissRoll" in self.hparams.data_name),
                    # 'Kmeans':e.E_Clasting_Kmeans(),
                    # 'metric/Dismatcher':e.E_Dismatcher(),
                })
                # with open(f"log/{now}/test.log", 'w') as f:
                #     f.write(str(result))
                #     f.write("\n")
                #     f.write(str(self.wandb_logs))
            except Exception as e:
                print(repr(e))
                print(label)
                print(latent)
                print(predictint)

            # e = eval_core.Eval(
            #     input=data,
            #     latent=latent,
            #     label=label,
            #     k=10
            #     )
            # self.wandb_logs['metric/AccSvc_dimall'] = e.E_Classifacation_SVC()

            for i in range(len(self.labelstr)):

                # if self.hparams.plotInput == 1:
                #     self.hparams.plotInput = 0
                #     if latent.shape[1] == 3:
                #         self.wandb_logs['visualize/Inputembdeing{}'.format(str(i))] = px.scatter_3d(
                #             x=data[:, 0], y=data[:, 1], z=data[:, 2],
                #             color=np.array(self.labelstr[i])[index],
                #             color_continuous_scale='speed'
                #         )

                if latent.shape[1] == 2:
                    self.wandb_logs['visualize/test_embdeing{}'.format(str(i))] = px.scatter(x=latent[:, 0], y=latent[:, 1], color=np.array(self.labelstr[i])[index.cpu()])
                    self.wandb_logs['visualize/test_valembdeing{}'.format(str(i))] = px.scatter(x=lattest[:, 0], y=lattest[:, 1], color=np.array(label))
                    # img.write_image(f"log/{now}/visualize/embdeing{str(i)}_{self.current_epoch}.png", engine="kaleido")
                    # img.write_image(f"log/{now}/visualize/testembdeing{str(i)}_{self.current_epoch}.png", engine="kaleido")
                    # self.wandb_logs['visualize/embdeing{}'.format(str(i))] = px.scatter(
                    #     x=latent[:, 0], y=latent[:, 1], color=np.array(self.labelstr[i])[index])
                    # self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = px.scatter(
                    #     x=lattest[:, 0], y=lattest[:, 1], color=np.array(labeltest))

                elif latent.shape[1] == 3:
                    self.wandb_logs['visualize/embdeing{}'.format(str(i))] = px.scatter_3d(
                        x=latent[:, 0], y=latent[:, 1], z=latent[:, 2], color=np.array(self.labelstr[i])[index])
                # df = pd.concat(
                #     [
                #         pd.DataFrame(latent, columns=['x','y']),
                #         pd.DataFrame(np.array(self.labelstr[i])[index], columns=['label'])],
                #         axis=1
                #     )
                # df.to_csv('tem/metadata_{}.csv'.format(self.current_epoch))
                # wandb.save('tem/metadata_{}.csv'.format(self.current_epoch))

            # if self.hparams.show_detail:
            #     self.wandb_logs['data/p'] = px.imshow(self.P)
            #     self.wandb_logs['data/q'] = px.imshow(self.Q)
            #     self.wandb_logs['data/disq'] = px.imshow(self.dis_Q)
            #     self.wandb_logs['data/select_index_near'] = px.imshow(self.P > 0.5)
            #     self.wandb_logs['data/select_index_far'] = px.imshow(self.P < 1e-9)

            # wandb.log(self.wandb_logs)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = StepLR(optimizer, step_size=self.hparams.epochs // 10, gamma=0.5)

        return [optimizer], [self.scheduler]

    def visualize_embdeing(self, latent, label):
        return px.scatter(x=latent[:, 0], y=latent[:, 1], color=label)

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.data_train = self.train_dataset
            self.data_val = self.train_dataset

            self.train_da = DataLoader(
                self.data_train,
                shuffle=True,
                batch_size=self.train_batchsize,
                # pin_memory=True,
                num_workers=5,
                persistent_workers=True,
            )
            self.val_da = DataLoader(
                self.data_val,
                batch_size=self.train_batchsize,
                # pin_memory=True,
                num_workers=5,
                persistent_workers=True,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = self.train_dataset
            self.test_da = DataLoader(
                self.data_test,
                batch_size=self.test_batchsize,
                # pin_memory=True,
                num_workers=5,
                persistent_workers=True,
            )
            # self.mnist_test = self.train_dataset

    def train_dataloader(self):
        return self.train_da

    def val_dataloader(self):
        return self.val_da

    def test_dataloader(self):
        return self.test_da
        # return DataLoader(self.mnist_test, batch_size=self.test_batchsize)

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):

        NetworkStructure_1[0] = functools.reduce(lambda x, y: x * y, self.dims)
        model_1 = nn.ModuleList()
        # model_1.append(nn.Flatten())
        for i in range(len(NetworkStructure_1) - 1):
            if i != len(NetworkStructure_1) - 2:
                model_1.append(nn.Linear(NetworkStructure_1[i], NetworkStructure_1[i + 1]))
                model_1.append(nn.BatchNorm1d(NetworkStructure_1[i + 1]))
                model_1.append(nn.LeakyReLU(0.1))
            else:
                model_1.append(nn.Linear(NetworkStructure_1[i], NetworkStructure_1[i + 1]))
                model_1.append(nn.BatchNorm1d(NetworkStructure_1[i + 1]))
                model_1.append(nn.LeakyReLU(0.1))

        model_2 = nn.ModuleList()
        NetworkStructure_2[0] = NetworkStructure_1[-1]
        for i in range(len(NetworkStructure_2) - 1):
            if i != len(NetworkStructure_2) - 2:
                model_2.append(nn.Linear(NetworkStructure_2[i], NetworkStructure_2[i + 1]))
                model_2.append(nn.BatchNorm1d(NetworkStructure_2[i + 1]))
                model_2.append(nn.LeakyReLU(0.1))
            else:
                model_2.append(nn.Linear(NetworkStructure_2[i], NetworkStructure_2[i + 1]))
                model_2.append(nn.BatchNorm1d(NetworkStructure_2[i + 1]))
                model_2.append(nn.LeakyReLU(0.1))

        model_3 = nn.ModuleList()
        model_3.append(nn.Linear(NetworkStructure_2[-1], self.hparams.num_latent_dim))
        # model_3.append(nn.BatchNorm1d(self.hparams.num_latent_dim))

        return model_1, model_2, model_3


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

    wandb.init(
        name=runname,
        project="DSML_Blood",
        entity="zangzelin",
        mode='offline' if bool(args.__dict__['offline']) else 'online',
        save_code=True,
        # log_model=False,
        tags=[args.__dict__['data_name'], args.__dict__['method']],
        config=args,
    )

    refs = {
            'KNN': (neighbors.KNeighborsRegressor(n_neighbors=3), neighbors.KNeighborsClassifier(n_neighbors=3)),
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

    train_data = dataset.data.cpu().numpy()
    val_data = dataset.dataval.cpu().numpy()
    test_data = dataset.datatest.cpu().numpy()
    train_label = dataset.label.cpu().numpy().astype(np.int32)
    val_label = dataset.labelval.cpu().numpy().astype(np.int32)
    test_label = dataset.labeltest.cpu().numpy().astype(np.int32)

    clf = refs[args.method][1]
    clf.fit(train_data, train_label)

    val_predict = clf.predict(val_data)
    val_predict[val_predict<0.5] = 0
    val_predict[val_predict>0.5] = 1
    val_fpr, val_tpr, thresholds = metrics.roc_curve(val_predict, val_label)
    val_score = metrics.accuracy_score(val_predict, val_label)
    val_auc = metrics.auc(val_fpr, val_tpr)
    
    test_predict = clf.predict(test_data)
    test_predict[test_predict<0.5] = 0
    test_predict[test_predict>0.5] = 1
    test_fpr, test_tpr, thresholds = metrics.roc_curve(test_predict, test_label)
    test_score = metrics.accuracy_score(test_predict, test_label)
    test_auc = metrics.auc(test_fpr, test_tpr)

    wandb.log({
        'val_acc': val_score,
        'test_acc': test_score,
        'val_auc': val_auc,
        'test_auc': test_auc,
    })


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

    parser.add_argument('--scale', type=int, default=30)
    parser.add_argument('--vs', type=float, default=1e-2)
    parser.add_argument('--ve', type=float, default=-1)
    parser.add_argument('--K', type=int, default=15)

    # train param
    parser.add_argument('--batch_size', type=int, default=2000, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    print(args.foldindex)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" if args.foldindex < 5 else "1"
    main(args)
