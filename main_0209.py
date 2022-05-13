#!/usr/bin/env python
import importlib
from io import FileIO
import model
import os
from numpy.core.fromnumeric import size

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.progress import tqdm_progress
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
from Loss.dmt_lossce import myloss_ce
import nuscheduler
import wandb
import eval.eval_core as eval_core
import plotly.express as px
import matplotlib.pylab as plt 
import pandas as pd

import load_data_f.dataset as datasetfunc
from load_data_f.dataset import BlodNoMissingDataModule
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import sys
import git

# import networkx as nx
import random

torch.set_num_threads(1)

class DMT_Model(pl.LightningModule):
    
    def __init__(
        self,
        dataset:BlodNoMissingDataModule,
        testdataset:BlodNoMissingDataModule,
        DistanceF,
        SimilarityF,
        data_dir='./data',
        **kwargs,
        ):

        super().__init__()
        self.save_hyperparameters()
        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = self.hparams.lr
        self.train_batchsize = self.hparams.batch_size
        self.test_batchsize = self.hparams.batch_size
        # self.save_path = save_path
        # self.num_latent_dim = 2
        self.traindataset = dataset
        self.trainlabelstr = dataset.label_str
        self.dims = dataset.data[0].shape
        print("self.dims: ", self.dims)

        self.testdataset = testdataset
        self.testlabelstr = testdataset.label_str

        self.log_interval = self.hparams.log_interval
        self.DistanceF = DistanceF
        self.SimilarityF = SimilarityF
        # self.hparams.NetworkStructure.append(
        #     self.hparams.num_latent_dim
        #     )
        self.model, self.model2, self.model3 = self.InitNetworkMLP(self.hparams.NetworkStructure_1,self.hparams.NetworkStructure_2)
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
        self.celoss=nn.CrossEntropyLoss()
        # self.celoss = nn.MSELoss()
        # self.celoss = myloss_ce()
        self.wandb_logs = {}
        self.wandb_test_logs = {}
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
            epoch_start=self.hparams.epochs//5,
            epoch_end=self.hparams.epochs*4//5,
            )
        self.l_shadular = nuscheduler.Nushadular(
            nu_start=0.000001,
            nu_end=100,
            epoch_start=self.hparams.epochs//5,
            epoch_end=self.hparams.epochs*4//5,
            )
        

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

        lat3[lat3 < 0] = 0
        return lat1, lat3

    def aug_near_mix(self, index, dataset, k=10):

        r = (torch.arange(start=0, end=index.shape[0])*k + torch.randint(low=1, high=k, size=(index.shape[0],))).cuda()
        random_select_near_index = dataset.neighbors_index[index][:,:k].reshape((-1,))[r].long()
        random_select_near_data2 = dataset.data[random_select_near_index]
        random_rate = torch.rand(size = (index.shape[0], 1)).cuda()
        return random_rate*random_select_near_data2+(1-random_rate)*dataset.data[index]

    def training_step(self, batch, batch_idx):
        # data1, data2, rho, sigma, label, index = batch
        index = batch

        data1 = self.traindataset.data[index]
        print("\n\n\ndata1: ", data1)
        data2 = self.aug_near_mix(index, self.traindataset, k=self.hparams.K)
        label1 = torch.tensor(self.traindataset.label).long().clone().detach()[index]
        label2 = torch.tensor(self.traindataset.label).long().clone().detach()[index]

        # rho = self.dataset.rho[index]
        # sigma = self.dataset.sigma[index]
        # label = np.array(self.dataset.label)[index.cpu().numpy()]
        # label = self.dataset.label[index]

        mid1, lat1 = self(data1)
        mid2, lat2 = self(data2)
        data = torch.cat([mid1, mid2])
        lat = torch.cat([lat1, lat2])
        label = torch.cat([label1, label2]).to(self.device)

        loss = self.Loss(
            input_data=data.reshape(data.shape[0], -1),
            latent_data=lat.reshape(data.shape[0], -1),
            rho=0.0, # torch.cat([rho, rho]),
            sigma=1.0, # torch.cat([sigma, sigma]),
            v_latent=self.nushadular.Getnu(self.current_epoch),
        ).float()
        # lat_loss = lat.reshape((-1,))
        # lat_loss[lat_loss < 0] = 0
        # loss_ce = self.celoss(lat.reshape((-1,)), label).float()
        loss_ce = self.celoss(lat, label).float()
        # print("--------------------------------------\nloss: {}, loss_ce: {}".format(loss, loss_ce))
        # print("lat shape: {}, label shape: {}, loss_ce: {}".format(lat.shape, label.shape, loss_ce))

        # self.logger.experiment.
        total_loss = loss + loss_ce / 50

        self.wandb_logs={
            'loss_train': loss,
            'loss_ce_train': loss_ce,
            'total_loss_train': total_loss,
            'nv': self.nushadular.Getnu(self.current_epoch),
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'dimention_loss': self.l_shadular.Getnu(self.current_epoch),
            'epoch': self.current_epoch,
        }
        # wandb.log(self.wandb_logs)

        # if self.hparams.NetworkStructure[-1] >2:
        #     loss += torch.mean(lat1[:, 2:]) * self.l_shadular.Getnu(self.current_epoch)
        # self.scheduler.step()
        self.log('total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        index = batch

        data = self.traindataset.data[index]
        data2 = self.aug_near_mix(index, self.traindataset, k=self.hparams.K)
        # data2 = self.dataset.data[index]
        # rho = self.dataset.rho[index]
        # sigma = self.dataset.sigma[index]
        label = np.array(self.traindataset.label)[index.cpu().numpy()]
        # latent = self.model(data)
        mid, lat = self(data)
        mid2, lat2 = self(data2)
                
        return (
            data.detach().cpu().numpy(),
            lat.detach().cpu().numpy(),
            lat2.detach().cpu().numpy(),
            label,
            index.detach().cpu().numpy(),
            )

    def validation_epoch_end(self, outputs):
        
        # print('self..current_epoch',self.current_epoch)
        if (self.current_epoch+1) % self.log_interval == 0:
            
            data = np.concatenate([ data_item[0] for data_item in outputs ])
            latent = np.concatenate([ data_item[1] for data_item in outputs ])
            latent2 = np.concatenate([ data_item[2] for data_item in outputs ])
            label = np.concatenate([ data_item[3] for data_item in outputs ])
            index = np.concatenate([ data_item[4] for data_item in outputs ])

            e = eval_core.Eval(
                input=data,
                latent=latent,
                label=label,
                k=10
                )

            midtest, lattest = self(self.traindataset.data.to(self.device))
            labeltest = Tensor(self.traindataset.label).long().to(self.device)

            # lattest = nn.Softmax(dim=1)(lattest)

            # print(lattest.reshape((-1,)).size())
            # print(tensor(labeltest).size())
            # lattest = self.celoss(lattest.reshape((-1,)).cpu(), tensor(labeltest))
            # lattest = lattest.cpu().numpy()
            
            # 分类任务
            # predictint = lattest[:,1].copy()
            # predictint[predictint>0.5] = 1
            # predictint[predictint<=0.5] = 0

            # 回归任务
            # predictint = lattest.copy()
            
            loss_ce = self.celoss(lattest, labeltest).float()

            self.wandb_logs.update({
                'epoch': self.current_epoch,
                'validation-mse': loss_ce,
            })
            # print(self.wandb_logs)
            
            for i in range(len(self.trainlabelstr)):
                
                if self.hparams.plotInput==1:
                    self.hparams.plotInput=0
                    if latent.shape[1] == 3:
                        self.wandb_logs['visualize/testInputembdeing{}'.format(str(i))] = px.scatter_3d(
                            x=data[:,0], y=data[:,1], z=data[:,2], 
                            color=np.array(self.trainlabelstr[i])[index],
                            color_continuous_scale='speed'
                            )  
                
                if latent.shape[1] == 2:
                    # self.wandb_logs['visualize/embdeing{}'.format(str(i))] = px.scatter(x=latent[:,0], y=latent[:,1], color=np.array(self.labelstr[i])[index])
                    # self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = px.scatter(x=lattest[:,0], y=lattest[:,1], color=np.array(labeltest))

                    self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing2d(latent, np.array(self.trainlabelstr[i])[index])
                    self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing2d(lattest.cpu().numpy(), labeltest.cpu().numpy())
                
                elif latent.shape[1] == 3:
                    # self.wandb_logs['visualize/embdeing{}'.format(str(i))] = px.scatter_3d(x=latent[:,0], y=latent[:,1], z=latent[:,2], color=np.array(self.labelstr[i])[index])
                    # self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing3d(latent, np.array(self.labeltest))
                    self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing3d(latent, np.array(self.trainlabelstr[i])[index])
                    self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing3d(lattest, np.array(self.trainlabelstr))

            wandb.log(self.wandb_logs)

    def test_step(self, batch, batch_idx):
        index = batch
        # print(index)

        data = self.testdataset.data[index]
        data2 = self.aug_near_mix(index, self.testdataset, k=self.hparams.K)
        # data2 = self.testdataset.data[index]
        # rho = self.testdataset.rho[index]
        # sigma = self.testdataset.sigma[index]
        label = np.array(self.testdataset.label)[index.cpu().numpy()]
        # latent = self.model(data)
        mid, lat = self(data)
        mid2, lat2 = self(data2)
        
        return (
            data.detach().cpu().numpy(),
            lat.detach().cpu().numpy(),
            lat2.detach().cpu().numpy(),
            label,
            index.detach().cpu().numpy(),
            )

    def test_epoch_end(self, outputs):
        
        # print('self..current_epoch',self.current_epoch)
        # if (self.current_epoch+1) % self.log_interval == 0:
            
        data = np.concatenate([ data_item[0] for data_item in outputs ])
        latent = np.concatenate([ data_item[1] for data_item in outputs ])
        latent2 = np.concatenate([ data_item[2] for data_item in outputs ])
        label = np.concatenate([ data_item[3] for data_item in outputs ])
        index = np.concatenate([ data_item[4] for data_item in outputs ])

        e = eval_core.Eval(
            input=data,
            latent=latent,
            label=label,
            k=10
            )

        # 分类任务
        datatest = self.testdataset.data
        midtest, lattest = self(datatest.to(self.device))
        labeltest = Tensor(self.testdataset.label.astype(np.int32)).long().to(self.device)
        # print(datatest, labeltest)

        loss_ce = self.celoss(lattest, labeltest).cpu()

        # 分类任务
        lattest = nn.Softmax()(lattest)
        lattest = lattest.cpu().numpy()
        
        predictint = lattest[:,1].copy()
        # print(predictint)
        predictint[predictint>0.5] = 1
        predictint[predictint<=0.5] = 0
        # print("predict: ", predictint)
        # print("truth: ", labeltest.cpu())
        # 回归任务
        # lattest = nn.MSELoss()(lattest.reshape((-1,)).cpu(), tensor(labeltest))
        # predictint = lattest.copy()

        # 回归
        # loss_ce = self.celoss(lattest.reshape((-1,)).cpu(), labeltest).float()
        # 分类
        # print("test loss ce: ", loss_ce)

        from sklearn import metrics

        # print("label: {}\nlat: {}\n".format(labeltest.cpu(), predictint))
        
        self.wandb_test_logs={
            'loss_ce_test': loss_ce,
            'metric/auc': metrics.roc_auc_score(labeltest.cpu(), lattest[:,1]),
            'metric/acc': metrics.accuracy_score(labeltest.cpu(), predictint),
            'epoch': self.current_epoch,
        }
        # self.result = 

        # print("test log: {}".format(self.wandb_test_logs))
        
        # for i in range(len(self.testlabelstr)):
            
        #     if self.hparams.plotInput==1:
        #         self.hparams.plotInput=0
        #         if latent.shape[1] == 3:
        #             self.wandb_test_logs['visualize/Inputembdeing{}'.format(str(i))] = px.scatter_3d(
        #                 x=data[:,0], y=data[:,1], z=data[:,2], 
        #                 color=np.array(self.testlabelstr[i])[index],
        #                 color_continuous_scale='speed'
        #                 )  
            
        #     if latent.shape[1] == 2:
        #         # self.wandb_test_logs['visualize/embdeing{}'.format(str(i))] = px.scatter(x=latent[:,0], y=latent[:,1], color=np.array(self.labelstr[i])[index])
        #         # self.wandb_test_logs['visualize/testembdeing{}'.format(str(i))] = px.scatter(x=lattest[:,0], y=lattest[:,1], color=np.array(labeltest))

        #         self.wandb_test_logs['visualize/embdeing{}'.format(str(i))] = self.visualize_embdeing2d(latent, np.array(self.testlabelstr[i])[index])
        #         self.wandb_test_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing2d(lattest.cpu().numpy(), np.array(labeltest.cpu()))
            
        #     elif latent.shape[1] == 3:
        #         # self.wandb_test_logs['visualize/embdeing{}'.format(str(i))] = px.scatter_3d(x=latent[:,0], y=latent[:,1], z=latent[:,2], color=np.array(self.labelstr[i])[index])
        #         # self.wandb_test_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing3d(latent, np.array(self.labeltest))
        #         self.wandb_test_logs['visualize/embdeing{}'.format(str(i))] = self.visualize_embdeing3d(latent, np.array(self.testlabelstr[i])[index])
        #         self.wandb_test_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing3d(lattest, np.array(labeltest))
            # df = pd.concat(
            #     [
            #         pd.DataFrame(latent, columns=['x','y']), 
            #         pd.DataFrame(np.array(self.labelstr[i])[index], columns=['label'])],
            #         axis=1
            #     )
            # df.to_csv('tem/metadata_{}.csv'.format(self.current_epoch))
            # wandb.save('tem/metadata_{}.csv'.format(self.current_epoch))
        
        # if self.hparams.show_detail:
        #     self.wandb_test_logs['data/p'] = px.imshow(self.P)
        #     self.wandb_test_logs['data/q'] = px.imshow(self.Q)
        #     self.wandb_test_logs['data/disq'] = px.imshow(self.dis_Q)
        #     self.wandb_test_logs['data/select_index_near'] = px.imshow(self.P > 0.5)
        #     self.wandb_test_logs['data/select_index_far'] = px.imshow(self.P < 1e-9)
        # wandb.log(self.wandb_test_logs)
            
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = StepLR(optimizer, step_size=self.hparams.epochs//10, gamma=0.5)
        
        return [optimizer], [self.scheduler]

    def visualize_embdeing2d(self, latent, label):
        return px.scatter(x=latent[:,0], y=latent[:,1], color=label)
    
    def visualize_embdeing3d(self, latent, label):
        return px.scatter(x=latent[:,0], y=latent[:,1], z=latent[:,2], color=label)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # self.data_train = self.dataset
            # self.data_val = self.dataset

            self.train_da = DataLoader(
                self.traindataset,
                shuffle=True,
                batch_size=self.train_batchsize,
                # pin_memory=True,
                num_workers=5,
                persistent_workers=True,
            )
            self.vali_da = DataLoader(
                self.traindataset,
                batch_size=self.train_batchsize,
                # pin_memory=True,
                num_workers=5,
                persistent_workers=True,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.testdataset = self.testdataset
            self.test_da = DataLoader(
                self.testdataset,
                batch_size=self.test_batchsize,
                # pin_memory=True,
                num_workers=5,
                persistent_workers=True,
            )

    def train_dataloader(self):
        return self.train_da

    def val_dataloader(self):
        return self.vali_da

    def test_dataloader(self):
        return self.test_da

    def InitNetworkMLP(self, NetworkStructure_1, NetworkStructure_2):

        NetworkStructure_1[0] = functools.reduce(lambda x,y:x * y, self.dims)
        model_1 = nn.ModuleList()
        # model_1.append(nn.Flatten())
        for i in range(len(NetworkStructure_1) - 1):
            if i != len(NetworkStructure_1) - 2:
                model_1.append(nn.Linear(NetworkStructure_1[i],NetworkStructure_1[i + 1]))
                model_1.append(nn.BatchNorm1d(NetworkStructure_1[i + 1]))
                model_1.append(nn.LeakyReLU(0.1))
            else:
                model_1.append(nn.Linear(NetworkStructure_1[i],NetworkStructure_1[i + 1]))
                model_1.append(nn.BatchNorm1d(NetworkStructure_1[i + 1]))
                model_1.append(nn.LeakyReLU(0.1))
            model_1.append(nn.Softmax(dim=1))
        
        model_2 = nn.ModuleList()
        NetworkStructure_2[0] = NetworkStructure_1[-1]
        for i in range(len(NetworkStructure_2) - 1):
            if i != len(NetworkStructure_2) - 2:
                model_2.append(nn.Linear(NetworkStructure_2[i],NetworkStructure_2[i + 1]))
                model_2.append(nn.BatchNorm1d(NetworkStructure_2[i + 1]))
                model_2.append(nn.LeakyReLU(0.1))
            else:
                model_2.append(nn.Linear(NetworkStructure_2[i],NetworkStructure_2[i + 1]))
                model_2.append(nn.BatchNorm1d(NetworkStructure_2[i + 1]))
                model_2.append(nn.LeakyReLU(0.1))
            model_2.append(nn.Softmax(dim=1))

        model_3 = nn.ModuleList()
        model_3.append(nn.Linear(NetworkStructure_2[-1], self.hparams.num_latent_dim))
        # model_3.append(nn.BatchNorm1d(self.hparams.num_latent_dim))

        return model_1, model_2, model_3

def main(args, train_data, train_label, test_data, test_label):
    early_stopping = EarlyStopping('total_loss', patience=50)
    # print(train_index.shape, test_index.shape)
    trainer = pl.Trainer.from_argparse_args(
        default_root_dir="checkpoints/2",
        args=args,
        gpus=1,
        max_epochs=args.epochs, 
        # enable_progress_bar=False,
        progress_bar_refresh_rate=10,
        # check_val_every_n_epoch=args.log_interval,
        logger=False,
        callbacks=[early_stopping]
        )
        
    disfunc_use = getattr(disfunc, 'EuclideanDistanceNumpy')
    simfunc_use = getattr(simfunc, 'UMAPSimilarity')
    simfunc_npuse = getattr(simfunc, 'UMAPSimilarityNumpy')
    dm_class = getattr(datasetfunc, args.__dict__['data_name'] + 'NoMissingDataModuleNew')

    dataset = dm_class(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        data=train_data,
        label=train_label,
        **args.__dict__,
        )
    # print(train_data)

    testdataset = dm_class(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        data=test_data,
        label=test_label,
        **args.__dict__,
        )

    model = DMT_Model(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        dataset=dataset,
        testdataset=testdataset,
        **args.__dict__,
        )
    # model.load_state_dict(torch.load('./checkpoints/result.pth'))

    trainer.fit(model)
    trainer.test(model)
    return "The {} fold, result: {}\n".format(key, model.wandb_test_logs)
    # f.write("The {} fold, result: {}\n".format(key, model.result))
    # results.append(model.result.cpu())
    # del(trainer)
    # del(model)
    # f.write("mean: {}, std: {}\n".format(np.mean(results), np.std(results)))
            # torch.save(model.state_dict(), './checkpoints/result.pth')


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='digits_T', )
    parser.add_argument('-num_workers', default=1, type=int)

    # data set param
    parser.add_argument('--data_name', type=str, default='Blod', 
                        choices=[
                            # 'Digits', 'Coil20', 'Coil100',
                            # 'Smile', 'ToyDiff', 'SwissRoll',
                            # 'KMnist', 'EMnist', 'Mnist',
                            # 'EMnistBC', 'EMnistBYCLASS',
                            # 'Cifar10', 'Colon', 'PeiHuman',
                            # 'Gast10k', 'HCL60K', 'PBMC', 
                            # 'HCL280K', 'SAMUSIK', 
                            # 'M_handwritten', 'Seversphere',
                            # 'MCA', 'Activity', 'SwissRoll2',
                            'BlodZHEER', 'BlodAll', 'BlodNoMissing',
                            ])
    parser.add_argument('--n_point', type=int, default=60000000, )
    # wandb config
    parser.add_argument('--username', type=str, default="sqma", )
    parser.add_argument('--project', type=str, default="DLME_manifold", )
    parser.add_argument('--offline', type=int, default=0)
    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--v_input', type=float, default=100)
    parser.add_argument('--same_sigma', type=bool, default=False)
    parser.add_argument('--show_detail', type=bool, default=False)
    parser.add_argument('--perplexity', type=int, default=20)
    parser.add_argument('--plotInput', type=int, default=0)
    
    parser.add_argument('--vs', type=float, default=1e-2)
    parser.add_argument('--ve', type=float, default=-1)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--NetworkStructure_1', type=list, default=[-1, 500, 300, 80])
    parser.add_argument('--NetworkStructure_2', type=list, default=[-1, 500, 80])
    # 分类
    parser.add_argument('--num_latent_dim', type=int, default=2)
    # 回归
    # parser.add_argument('--num_latent_dim', type=int, default=1)
    # parser.add_argument('--NetworkStructure', type=list, default=[-1, 5000, 4000, 3000, 2000, 1000, 2])
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--pow_input', type=float, default=2)
    parser.add_argument('--K', type=int, default=15)
    parser.add_argument('--pow_latent', type=float, default=2)
    parser.add_argument('--method', type=str, default='dmt',
                        choices=['dmt', 'dmt_mask'])
    parser.add_argument('--augNearRate', type=float, default=100)

    parser.add_argument('--near_bound', type=float, default=0.0)
    parser.add_argument('--far_bound', type=float, default=1.0)
    parser.add_argument('--foldindex', type=int, default=1)

    # train param
    parser.add_argument('--batch_size', type=int, default=300, )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    
    pl.utilities.seed.seed_everything(args.__dict__['seed'])
    info = [str(s) for s in sys.argv[1:]]
    runname = '_'.join(['dmt', args.data_name, 'nNull', ''.join(info)])
    
    wandb.init(
        name = runname,
        project = args.__dict__['project'],
        entity = args.__dict__['username'],
        mode= 'offline' if bool(args.__dict__['offline']) else 'online',
        save_code=True,
        # log_model=False,
        tags=[args.__dict__['data_name'], args.__dict__['method']],
        config=args,
    )

    # from loaddata import preprocess
    # (
    #     nNull, missing, fillWithMean, fillWithMiddle, fillWithKNN,
    #  ) = preprocess(knn_neighbors=2, ifNorm=True, preSave=True)
    from loaddata import ReadRawData, SelectNoMissingData, DataFileWithMean, DataFileWithMedian
    nNullData = SelectNoMissingData(ReadRawData())
    data = pd.concat([nNullData.loc[:, 'heart_rate':'albumin'].apply(lambda x: (
        x - np.min(x)) / (np.max(x) - np.min(x))), nNullData.loc[:, 'upperbody':'c3']], axis=1).values
    label = nNullData.blood.copy().values.astype(int)
    # label[label > 0] = 1
    label[label > 0.5] = 1
    label[label <= 0.5] = 0

    # data, label = nNull['data'], nNull['label']
    # data, label = fillWithMean['data'], fillWithMean['label']
    # data, label = fillWithMiddle['data'], fillWithMiddle['label']
    nfold = 10
    
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=10, random_state=2022, shuffle=True)

    # logfile = str(args.foldindex) + "_fold.log"
    logfile = "fold.log"

    rr = []
    target = label.copy()
    # 分类任务
    target[target > 0.5] = 1
    target[target <= 0.5] = 0
    target = target.astype(np.int32)
    # print(data, '\n', target)
    # print(target)

    # with open(logfile, 'w+') as f:
    #     for key, (train_index, test_index) in enumerate(kf.split(data, target)):
            # if key >= args.foldindex * (nfold / torch.cuda.device_count()) and key < (args.foldindex + 1) * (nfold / torch.cuda.device_count()):
            # if key == args.foldindex:
            # mp.spawn(main, nprocs=nfold, args=(args, f, nfold))
    for key, (train_index, test_index) in enumerate(kf.split(data, target)):
        if key == args.foldindex:
            # print(args.foldindex, test_index)
            r = main(args, data[train_index], label[train_index], data[test_index], target[test_index])
            print(r)
    #         rr.append(r)
    
    # print(rr)
