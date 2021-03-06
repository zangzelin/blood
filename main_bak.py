#!/usr/bin/env python
from torch import tensor
import model
import os
from numpy.core.fromnumeric import size

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
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
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import sys
import git

import networkx as nx
import random

torch.set_num_threads(1)

class DMT_Model(pl.LightningModule):
    
    def __init__(
        self,
        traindataset,
        testdataset,
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
        self.test_batchsize = self.hparams.test_batch_size
        # self.test_batchsize = self.hparams.batch_size
        # self.save_path = save_path
        # self.num_latent_dim = 2
        self.dims = traindataset.data[0].shape
        self.log_interval = self.hparams.log_interval

        self.train_dataset = traindataset
        self.test_dataset = testdataset
        self.DistanceF = DistanceF
        self.SimilarityF = SimilarityF
        # self.hparams.NetworkStructure.append(
        #     self.hparams.num_latent_dim
        #     )
        self.model, self.model2, self.model3 = self.InitNetworkMLP(self.hparams.NetworkStructure_1,self.hparams.NetworkStructure_2)
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
        # self.celoss=nn.CrossEntropyLoss()
        # self.celoss = nn.MSELoss()
        self.celoss = myloss_ce()
        self.wandb_logs = {}
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
        
        self.labelstr = traindataset.label_str


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

        r = (torch.arange(start=0, end=index.shape[0])*k+torch.randint(low=1, high=k, size=(index.shape[0],))).cuda()
        # r = (torch.arange(start=0, end=index.shape[0])*k+torch.randint(low=1, high=k, size=(index.shape[0],)))
        random_select_near_index = dataset.neighbors_index[index][:,:k].reshape((-1,))[r].long()
        # print("index: ", index)
        # print(torch.arange(start=0, end=index.shape[0])*k)
        # print(torch.randint(low=1, high=k, size=(index.shape[0],)))
        # print(r)
        # print(dataset.neighbors_index.shape)
        # print("random_select_near_index_train: ", random_select_near_index)
        # print(dataset.data.shape)
        random_select_near_data2 = dataset.data[random_select_near_index]
        random_rate = torch.rand(size = (index.shape[0], 1)).cuda()
        # random_rate = torch.rand(size = (index.shape[0], 1))
        return random_rate * random_select_near_data2 + (1 - random_rate) * dataset.data[index]
    
    def aug_near_mix_test(self, index, dataset, k=10):
        r = (torch.arange(start=0, end=index.shape[0])*k+torch.randint(low=1, high=k, size=(index.shape[0],))).cuda()
        # r = (torch.arange(start=0, end=index.shape[0])*k+torch.randint(low=1, high=k, size=(index.shape[0],)))
        random_select_near_index = dataset.neighbors_indextest[index][:,:k].reshape((-1,))[r].long()
        # print("index: ", index)
        # print(torch.arange(start=0, end=index.shape[0])*k)
        # print(torch.randint(low=1, high=k, size=(index.shape[0],)))
        # print(r)
        # print(dataset.neighbors_indextest.shape)
        # print("random_select_near_index_test: ", random_select_near_index)
        # print(dataset.datatest.shape)
        random_select_near_data2 = dataset.datatest[random_select_near_index]
        random_rate = torch.rand(size = (index.shape[0], 1)).cuda()
        # random_rate = torch.rand(size = (index.shape[0], 1))
        return random_rate * random_select_near_data2 + (1 - random_rate) * dataset.datatest[index]

    def training_step(self, batch, batch_idx):
        # data1, data2, rho, sigma, label, index = batch
        index = batch

        data1 = self.data_train.data[index]
        data2 = self.aug_near_mix(index, self.data_train, k=self.hparams.K)
        # print(type(self.data_train.label.dtype))
        label1 = torch.tensor(self.data_train.label).clone().detach().requires_grad_(True)[index]
        label2 = torch.tensor(self.data_train.label).clone().detach().requires_grad_(True)[index]

        # rho = self.data_train.rho[index]
        # sigma = self.data_train.sigma[index]
        # label = np.array(self.data_train.label)[index.cpu().numpy()]
        # label = self.data_train.label[index]

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
        loss_ce = self.celoss(lat.reshape((-1,)), label).float()
        # print("loss: {}, loss_ce: {}".format(loss.dtype, loss_ce.dtype))
        # print("lat shape: {}, label shape: {}, loss_ce: {}".format(lat.shape, label.shape, loss_ce))

        # self.logger.experiment.
        total_loss = loss + loss_ce

        self.wandb_logs={
            'loss_train': loss,
            'loss_ce_train': loss_ce,
            'total_loss_train': total_loss,
            'nv': self.nushadular.Getnu(self.current_epoch),
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            'dimention_loss': self.l_shadular.Getnu(self.current_epoch),
            'epoch': self.current_epoch,
        }
        # wandb.log(logs)

        # if self.hparams.NetworkStructure[-1] >2:
        #     loss += torch.mean(lat1[:, 2:]) * self.l_shadular.Getnu(self.current_epoch)
        # self.scheduler.step()
        self.log('total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        index = batch

        # data = self.data_train.data[index]
        data = self.data_val.data[index]
        data2 = self.aug_near_mix(index, self.data_train, k=self.hparams.K)
        # data2 = self.data_train.data[index]
        # rho = self.data_train.rho[index]
        # sigma = self.data_train.sigma[index]
        label = np.array(self.data_val.label)[index.cpu().numpy()]
        # latent = self.model(data)
        mid, lat = self(data)
        mid2, lat2 = self(data2)
        
        # self.loss_ce, self.dis_P, self.dis_Q, self.P, self.Q = self.Loss.ForwardInfo(
        #     input_data=data.reshape(data.shape[0], -1),
        #     latent_data=lat.reshape(data.shape[0], -1),
        #     rho=0.0,
        #     sigma=1.0,
        #     v_latent=self.nushadular.Getnu(self.current_epoch),
        # )
        # s_index = np.argsort(label.detach().cpu().numpy())[:500]
        # self.P = self.P[s_index,:][:,s_index]
        # self.Q = self.Q[s_index,:][:,s_index]
        # self.dis_P = self.dis_P[s_index,:][:,s_index]
        # self.dis_Q = self.dis_Q[s_index,:][:,s_index]
        # index = np.eye(self.P.shape[0], dtype=np.bool)
        # self.P[index] = np.NaN
        # self.Q[index] = np.NaN
        
        return (
            data.detach().cpu().numpy(),
            lat.detach().cpu().numpy(),
            lat2.detach().cpu().numpy(),
            label,
            index.detach().cpu().numpy(),
            )

    def test_step(self, batch, batch_idx):
        print("index: ", batch)
        index = batch

        # data = self.data_train.data[index]
        data1 = self.data_test.datatest[index]
        # data2 = self.data_test.datatest[index]
        data2 = self.aug_near_mix_test(index, self.data_test, k=self.hparams.K)
        # # data2 = self.data_train.data[index]
        # # rho = self.data_train.rho[index]
        # # sigma = self.data_train.sigma[index]
        # label = tensor(self.data_test.label)[index]
        label1 = tensor(self.data_test.labeltest)[index]
        label2 = tensor(self.data_test.labeltest)[index]
        # # latent = self.model(data)
        # mid, lat = self(data)
        # mid2, lat2 = self(data2)

        # data1 = self.data_train.data[index]
        # data2 = self.aug_near_mix(index, self.data_train, k=self.hparams.K)
        # # print(type(self.data_train.label.dtype))
        # label1 = torch.tensor(self.data_train.label)[index]
        # label2 = torch.tensor(self.data_train.label)[index]

        # rho = self.data_train.rho[index]
        # sigma = self.data_train.sigma[index]
        # label = np.array(self.data_train.label)[index.cpu().numpy()]
        # label = self.data_train.label[index]

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
        loss_ce = self.celoss(lat.reshape((-1,)), label).float()
        # print("loss: {}, loss_ce: {}".format(loss.dtype, loss_ce.dtype))
        # print("lat shape: {}, label shape: {}, loss_ce: {}".format(lat.shape, label.shape, loss_ce))

        # self.logger.experiment.
        total_loss = loss + loss_ce
        
        self.wandb_logs={
            'loss_test': loss,
            'loss_ce_test': loss_ce,
            'total_loss_test': total_loss,
            'epoch_test': self.current_epoch,
        }

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

            datatest = self.data_train.datatest
            midtest, lattest = self(datatest.to(self.device))
            labeltest = self.data_train.labeltest.astype(np.int32)

            lattest = nn.Softmax(dim=1)(lattest)
            lattest = lattest.cpu().numpy()
            
            # ????????????
            # predictint = lattest[:,1].copy()
            # predictint[predictint>0.5] = 1
            # predictint[predictint<=0.5] = 0

            # ????????????
            predictint = lattest.copy()

            # if self.path_list == None:
            from sklearn import metrics
            # fpr, tpr, thresholds = metrics.roc_curve(
            #     lattest[:,0], labeltest
            #     )

            self.wandb_logs.update({
                'epoch': self.current_epoch,
                # 'metric/auc': metrics.roc_auc_score(labeltest, lattest),
                # 'metric/acc': metrics.accuracy_score(labeltest, predictint)
                'metric/acc': (labeltest, predictint)
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
        
            # e = eval_core.Eval(
            #     input=data,
            #     latent=latent,
            #     label=label,
            #     k=10
            #     )
            # self.wandb_logs['metric/AccSvc_dimall'] = e.E_Classifacation_SVC() 
            
            for i in range(len(self.labelstr)):
                
                if self.hparams.plotInput==1:
                    self.hparams.plotInput=0
                    if latent.shape[1] == 3:
                        self.wandb_logs['visualize/Inputembdeing{}'.format(str(i))] = px.scatter_3d(
                            x=data[:,0], y=data[:,1], z=data[:,2], 
                            color=np.array(self.labelstr[i])[index],
                            color_continuous_scale='speed'
                            )  
                
                if latent.shape[1] == 2:
                    self.wandb_logs['visualize/embdeing{}'.format(str(i))] = self.visualize_embdeing2d(latent, np.array(self.labelstr[i])[index])
                    self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing2d(lattest, np.array(labeltest))
                
                elif latent.shape[1] == 3:
                    self.wandb_logs['visualize/embdeing{}'.format(str(i))] = self.visualize_embdeing3d(latent, np.array(self.labelstr[i])[index])
                    self.wandb_logs['visualize/testembdeing{}'.format(str(i))] = self.visualize_embdeing3d(lattest, np.array(self.labeltest))
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
            
            wandb.log(self.wandb_logs)

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
            self.data_train = self.train_dataset
            # self.data_val = self.train_dataset

            self.train_da = DataLoader(
                self.data_train,
                shuffle=True,
                batch_size=self.train_batchsize,
                # pin_memory=True,
                num_workers=5,
                persistent_workers=True,
            )
            # self.vali_da = DataLoader(
            #     self.data_val,
            #     batch_size=self.train_batchsize,
            #     # pin_memory=True,
            #     num_workers=5,
            #     persistent_workers=True,
            # )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.data_test = self.test_dataset
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

    # def val_dataloader(self):
    #     return self.vali_da

    def test_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=self.test_batchsize)
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
    train_dm_class = getattr(datasetfunc, args.__dict__['data_name'] + 'DataModuleTrain')
    test_dm_class = getattr(datasetfunc, args.__dict__['data_name'] + 'DataModuleTest')
    train_dataset = train_dm_class(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        **args.__dict__,
        )
    test_dataset = test_dm_class(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        **args.__dict__,
        )

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
    
    model = DMT_Model(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        traindataset=train_dataset,
        testdataset=test_dataset,
        **args.__dict__,
        )

    early_stopping = EarlyStopping('total_loss', patience=5)
    trainer = pl.Trainer.from_argparse_args(
        args=args,
        gpus=1,
        max_epochs=args.epochs, 
        progress_bar_refresh_rate=10,
        # check_val_every_n_epoch=args.log_interval,
        logger=False,
        callbacks=[early_stopping]
        )
    trainer.fit(model)
    # trainer.test()


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='digits_T', )
    parser.add_argument('-num_workers', default=1, type=int)

    # wandb config
    parser.add_argument('--username', type=str, default="sqma", )
    parser.add_argument('--project', type=str, default="DLME_manifold", )
    # data set param
    parser.add_argument('--data_name', type=str, default='BlodNoMissing', 
                        choices=[
                            # 'Digits', 'Coil20', 'Coil100',
                            # 'Smile', 'ToyDiff', 'SwissRoll',
                            # 'KMnist', 'EMnist', 'Mnist',
                            # 'EMnistBC', 'EMnistBYCLASS',
                            # 'Cifar10', 'Colon',
                            # 'Gast10k', 'HCL60K', 'PBMC', 
                            # 'HCL280K', 'SAMUSIK', 
                            # 'M_handwritten', 'Seversphere',
                            # 'MCA', 'Activity', 'SwissRoll2',
                            # 'PeiHuman', 'BlodZHEER', 'BlodAll' 
                            'BlodNoMissing',
                            ])
    parser.add_argument('--n_point', type=int, default=60000000, )
    # wandb config
    parser.add_argument('--username', type=str, default="sqma", )
    parser.add_argument('--project', type=str, default="DLME_manifold", )
    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--v_input', type=float, default=100)
    parser.add_argument('--same_sigma', type=bool, default=False)
    parser.add_argument('--show_detail', type=bool, default=False)
    parser.add_argument('--perplexity', type=int, default=10)
    parser.add_argument('--plotInput', type=int, default=0)
    
    parser.add_argument('--vs', type=float, default=1e-2)
    parser.add_argument('--ve', type=float, default=-1)
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--NetworkStructure_1', type=list, default=[-1, 500, 300, 80])
    parser.add_argument('--NetworkStructure_2', type=list, default=[-1, 500, 80])
    parser.add_argument('--num_latent_dim', type=int, default=1)
    # parser.add_argument('--NetworkStructure', type=list, default=[-1, 5000, 4000, 3000, 2000, 1000, 2])
    parser.add_argument('--model_type', type=str, default='mlp')
    parser.add_argument('--pow_input', type=float, default=2)
    parser.add_argument('--K', type=int, default=15)
    parser.add_argument('--pow_latent', type=float, default=2)
    parser.add_argument('--method', type=str, default='dmt',
                        choices=['dmt', 'dmt_mask'])
    parser.add_argument('--augNearRate', type=float, default=100)
    parser.add_argument('--offline', type=int, default=0)

    parser.add_argument('--near_bound', type=float, default=0.0)
    parser.add_argument('--far_bound', type=float, default=1.0)
    parser.add_argument('--foldindex', type=int, default=1)

    # train param
    parser.add_argument('--batch_size', type=int, default=300, )
    parser.add_argument('--test_batch_size', type=int, default=50, )
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()
    
    main(args)
