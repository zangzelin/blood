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
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_curve, average_precision_score

# import Similarity.sim_tdis as Sim_use
import Similarity.sim_Gaussian_1 as Sim_use
import Loss.dmt_loss_old as Loss_use
import Loss.dmt_loss_nearfar as Loss_use_mask
import nuscheduler
# import wandb
import eval.eval_core as eval_core
import plotly.express as px
import matplotlib.pylab as plt
import pandas as pd
from sklearn import metrics

import load_data_f.dataset as datasetfunc
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import sys
import git
from sklearn import metrics

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
        self.mseloss = torch.nn.MSELoss()
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
        self.model, self.model2, self.model3 = self.InitNetworkMLP(
            self.hparams.NetworkStructure_1, self.hparams.NetworkStructure_2)
        # print(self.hparams.NetworkStructure)
        # print(self.model)
        # print(self.model2)
        # print(self.model3)
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
        # self.l_shadular = nuscheduler.Nushadular(
        #     nu_start=0.000001,
        #     nu_end=100,
        #     epoch_start=self.hparams.epochs // 5,
        #     epoch_end=self.hparams.epochs * 4 // 5,
        # )

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
        if self.hparams.classfication_model == 1:
            loss_ce = self.celoss(lat, label.long())
        else:
            nn_softmax = nn.Softmax(dim=1)
            lab_learn = torch.cat(
                [(1 - label).reshape(-1, 1), label.reshape(-1, 1)],
                dim=1
                ).float()
            loss_ce = self.mseloss(nn_softmax(lat), lab_learn)
        # else:
        #     loss_ce = self.celoss(lat, label.long())
        total_loss = loss + loss_ce / self.hparams.scale

        # self.logger.experiment.
        self.wandb_logs = {
            'loss': loss,
            'loss_ce': loss_ce,
            'total_loss': total_loss,
            # 'nv': self.nushadular.Getnu(self.current_epoch),
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            # 'dimention_loss': self.l_shadular.Getnu(self.current_epoch),
            'epoch': self.current_epoch,
            # 'visualize/train_embdeing': px.scatter(x=lat1[:, 0].cpu().detach().numpy(), y=lat1[:, 1].cpu().detach().numpy(), color=label1.cpu().detach().numpy())
            # 'visualize/train_valembdeing' : px.scatter(x=lattest[:, 0], y=lattest[:, 1], color=np.array(label))
        }
        # print(self.wandb_logs)
        # wandb.log(self.wandb_logs)

        # if self.hparams.NetworkStructure[-1] >2:
        #     loss += torch.mean(lat1[:, 2:]) * self.l_shadular.Getnu(self.current_epoch)
        # self.scheduler.step()
        return total_loss

    def validation_step(self, batch, batch_idx):
        # print("batchsize: {}, shape:{}".format(batch.shape[0] , self.data_train.datatest.shape[0]))
        # index = batch if batch.shape[0] < self.data_val.dataval.shape[0] else torch.Tensor(range(self.data_val.dataval.shape[0])).long()
        # # print(batch, torch.Tensor(range(self.data_val.dataval.shape[0])).long())

        # # print(index)
        # data1 = self.data_val.dataval[index].to(self.device)
        # # data2 = self.aug_near_mix(index, self.data_val, stage="val", k=self.hparams.K).to(self.device)
        # # data2 = self.data_val.data[index]
        # # rho = self.data_val.rho[index]
        # # sigma = self.data_val.sigma[index]
        # label1 = np.array(self.data_val.labelval)
        # ind = index.cpu()
        # ind = ind.numpy()
        # label1 = label1[ind]
        # # label2 = np.array(self.data_val.labelval)[index.cpu().numpy()]
        # # latent = self.model(data)
        # mid1, lat1 = self(data1)
        # # mid2, lat2 = self(data2)

        # # data = torch.cat([mid1, mid2]).to(self.device)
        # # lat = torch.cat([lat1, lat2]).to(self.device)
        # # label = torch.cat([label1, label2]).to(self.device)
        # data = mid1.to(self.device)
        # lat = lat1.to(self.device)
        # label = torch.Tensor(label1).to(self.device)
        pass
        # return (
        #     data,
        #     lat,
        #     # lat2.detach().cpu().numpy(),
        #     label,
        #     index,
        # )

    def validation_epoch_end(self, outputs):

        if self.current_epoch == (self.hparams.epochs - 1):

            train_data = self.train_dataset.data.cpu().numpy()
            train_data = self.train_dataset.data.to(self.device)
            val_data = self.train_dataset.dataval.to(self.device)
            test_data = self.train_dataset.datatest.to(self.device)
            # train_label = self.train_dataset.label.cpu().numpy().astype(np.int32)
            # val_label = self.train_dataset.labelval.cpu().numpy().astype(np.int32)
            # test_label = self.train_dataset.labeltest.cpu().numpy().astype(np.int32)

            train_lat, train_emb = self(train_data)
            val_lat, val_emb = self(val_data)
            test_lat, test_emb = self(test_data)

            nn_softmax = nn.Softmax(dim=1)
            train_emb = nn_softmax(train_emb)
            val_emb = nn_softmax(val_emb)
            test_emb = nn_softmax(test_emb)

            if self.hparams.classfication_model == 1:
                train_label = self.train_dataset.label.cpu().numpy().astype(np.int32)
                val_label = self.train_dataset.labelval.cpu().numpy().astype(np.int32)
                test_label = self.train_dataset.labeltest.cpu().numpy().astype(np.int32)

                train_predict = (train_emb[:, 0] < train_emb[:, 1]).cpu().numpy().astype(int)
                val_predict = (val_emb[:, 0] < val_emb[:, 1]).cpu().numpy().astype(int)
                test_predict = (test_emb[:, 0] < test_emb[:, 1]).cpu().numpy().astype(int)

                train_fpr, train_tpr, thresholds = metrics.roc_curve(train_label, train_emb[:, 1].cpu().numpy())
                train_score = metrics.accuracy_score(train_predict, train_label)
                train_auc = metrics.auc(train_fpr, train_tpr)

                val_fpr, val_tpr, thresholds = metrics.roc_curve(val_label, val_emb[:, 1].cpu().numpy())
                val_score = metrics.accuracy_score(val_predict, val_label)
                val_auc = metrics.auc(val_fpr, val_tpr)
                
                test_fpr, test_tpr, thresholds = metrics.roc_curve(test_label, test_emb[:, 1].cpu().numpy())
                test_score = metrics.accuracy_score(test_predict, test_label)
                test_auc = metrics.auc(test_fpr, test_tpr)

                self.log_dict = {
                    'train_acc': train_score,
                    'val_acc': val_score,
                    'test_acc': test_score,
                    'train_auc': train_auc,
                    'val_auc': val_auc,
                    'test_auc': test_auc,
                }

            else:

                train_label = self.train_dataset.label.cpu().numpy()
                val_label = self.train_dataset.labelval.cpu().numpy()
                test_label = self.train_dataset.labeltest.cpu().numpy()

                nn_softmax = nn.Softmax(dim=1)
                train_predict = nn_softmax(train_emb)[:, 1].cpu().numpy()
                val_predict = nn_softmax(val_emb)[:, 1].cpu().numpy()
                test_predict = nn_softmax(test_emb)[:, 1].cpu().numpy()

                train_mse = mean_squared_error(train_label, train_predict)
                train_r = r2_score(train_label, train_predict)

                val_mse = mean_squared_error(val_label, val_predict)
                val_r = r2_score(val_label, val_predict)

                test_mse = mean_squared_error(test_label, test_predict)
                test_r = r2_score(test_label, test_predict)

                self.log_dict = {
                    'train_r': train_r,
                    'val_r': val_r,
                    'test_r': test_r,
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'test_mse': test_mse,
                }
            # wandb.log(self.log_dict)
        else:
            pass

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.weight_decay)
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

    # wandb.init(
    #     name=runname,
    #     project="DSML_Blood",
    #     entity="zangzelin",
    #     mode='offline' if bool(args.__dict__['offline']) else 'online',
    #     save_code=True,
    #     # log_model=False,
    #     tags=[args.__dict__['data_name'], args.__dict__['method']],
    #     config=args,
    # )
    callbacks_list = []
    model = DMT_Model(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        dataset=dataset,
        **args.__dict__,
    )

    # early_stopping = EarlyStopping('total_loss', patience=50)
    trainer = pl.Trainer.from_argparse_args(
        # default_root_dir="checkpoints/1",
        args=args,
        gpus=1,
        max_epochs=args.epochs,
        logger=False,
        callbacks=callbacks_list,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        # callbacks=[early_stopping]
    )
    trainer.fit(model)

    return model.log_dict
    # trainer.test(model)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='digits_T', )

    # data set param
    parser.add_argument('--data_name', type=str, default='BlodNNull',)

    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--v_input', type=float, default=100)
    parser.add_argument('--perplexity', type=int, default=20)
    parser.add_argument('--NetworkStructure_1', type=list, default=[-1, 500, 300, 80])
    parser.add_argument('--NetworkStructure_2', type=list, default=[-1, 500, 80])
    parser.add_argument('--num_latent_dim', type=int, default=2)
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--fill_set', type=str, default='nNull')
    parser.add_argument('--augNearRate', type=float, default=100)
    parser.add_argument('--offline', type=int, default=0)
    parser.add_argument('--method', type=str, default='dmt',
                        choices=['dmt', 'dmt_mask'])
    parser.add_argument('--foldindex', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--uselabel", type=int, default=0)

    parser.add_argument('--scale', type=int, default=30)
    parser.add_argument('--vs', type=float, default=1e-2)
    parser.add_argument('--ve', type=float, default=-1)
    parser.add_argument('--K', type=int, default=15)
    parser.add_argument("--classfication_model", type=int, default=0)

    # train param
    parser.add_argument('--batch_size', type=int, default=2000, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    # print(args.foldindex)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" if args.foldindex < 5 else "1"
    main(args)
