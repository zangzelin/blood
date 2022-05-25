import os
import joblib
import torch
import torchvision.datasets as datasets
from sklearn.metrics import pairwise_distances
import numpy as np
from load_data_f.sigma import PoolRunner
import scipy
from sklearn.preprocessing import StandardScaler
from pynndescent import NNDescent


class Source(torch.utils.data.Dataset):

    def __init__(self, DistanceF, SimilarityF, SimilarityNPF, foldindex, fill_set, uselabel, classfication_model, jumpPretreatment=False, **kwargs):
        self.args = kwargs
        self.DistanceF = DistanceF
        self.SimilarityF = SimilarityF
        self.SimilarityNPF = SimilarityNPF
        self.train = True
        self.foldindex = foldindex
        self.fill_set = fill_set
        self.classfication_model = classfication_model
        self._LoadData()
        # if not jumpPretreatment:
        # filename = 'data_name{}same_sigma{}perplexity{}v_input{}metric{}pow_input{}n_point{}'.format(
        #     self.args['data_name'],
        #     self.args['same_sigma'],
        #     self.args['perplexity'],
        #     self.args['v_input'],
        #     self.args['metric'],
        #     self.args['pow_input'],
        #     self.args['n_point'],
        # )
        if uselabel == 1:
            # print('----------------------------------')
            # print('use label find the neighbor')
            # print('----------------------------------')
            X_rshaped = self.data.reshape((self.data.shape[0], -1))
            # index = NNDescent(X_rshaped, n_jobs=-1, metric=self.args['metric'])
            # self.neighbors_index, neighbors_dist = index.query(X_rshaped, k=self.args['K'])
            dis = pairwise_distances(X_rshaped)

            if (self.label == 1).sum() > 0.1 * self.label.shape[0]:
                label_mask = (self.label * 10).int()
            else:
                label_mask = self.label.int()

            M = np.repeat(label_mask.reshape(1, -1), X_rshaped.shape[0], axis=0)
            dis[(M - M.T) != 0] = dis.max() + 1
            neighbors_index = dis.argsort(axis=1)[:, 1: self.args['K'] + 1]
            self.neighbors_index = torch.tensor(neighbors_index).cuda()
            self.data = self.data.reshape((self.data.shape[0], -1)).cuda()

            # X_rshapedval = self.dataval.reshape((self.dataval.shape[0], -1))
            # indexval = NNDescent(X_rshapedval, n_jobs=-1, metric=self.args['metric'])
            # self.neighbors_index_val, neighbors_distval = indexval.query(X_rshapedval, k=self.args['K'])
            # self.neighbors_index_val = torch.tensor(self.neighbors_index_val)
            self.dataval = self.dataval.reshape((self.dataval.shape[0], -1)).cuda()        

        else:
            X_rshaped = self.data.reshape((self.data.shape[0], -1))
            index = NNDescent(X_rshaped, n_jobs=-1, metric=self.args['metric'])
            self.neighbors_index, neighbors_dist = index.query(X_rshaped, k=self.args['K'])
            self.neighbors_index = torch.tensor(self.neighbors_index).cuda()
            self.data = self.data.reshape((self.data.shape[0], -1)).cuda()

            X_rshapedval = self.dataval.reshape((self.dataval.shape[0], -1))
            indexval = NNDescent(X_rshapedval, n_jobs=-1, metric=self.args['metric'])
            self.neighbors_index_val, neighbors_distval = indexval.query(X_rshapedval, k=self.args['K'])
            self.neighbors_index_val = torch.tensor(self.neighbors_index_val)
            self.dataval = self.dataval.reshape((self.dataval.shape[0], -1)).cuda()

        # if not os.path.exists('save/'+filename):
        # self._Pretreatment()
        #     joblib.dump(
        #         value=[self.sigma, self.rho, self.inputdim],
        #         filename='save/'+filename
        #         )
        # else:
        #     self.sigma, self.rho, self.inputdim = joblib.load('save/'+filename)

    def _LoadData(self, ):
        pass

    def _Pretreatment(self, ):

        if self.data.shape[0] > 0:
            rho, sigma = self._initKNN(
                self.data,
                perplexity=self.args['perplexity'],
                v_input=self.args['v_input']
            )
        else:
            # rho, sigma = self._initPairwiseMahalanobis(
            rho, sigma = self._initPairwise(
                self.data,
                perplexity=self.args['perplexity'],
                v_input=self.args['v_input'])

        self.sigma = torch.tensor(sigma).cuda()
        self.rho = torch.tensor(rho).cuda()
        self.inputdim = self.data[0].shape

    def _initPairwise(self, X, perplexity, v_input):
        print('use pairwise mehtod to find the sigma')

        dist = np.power(
            pairwise_distances(
                X.reshape((X.shape[0], -1)),
                n_jobs=-1,
                metric=self.args['metric']
            ),
            2,
        )
        rho = self._CalRho(dist)

        r = PoolRunner(
            similarity_function_nunpy=self.SimilarityNPF,
            number_point=X.shape[0],
            perplexity=perplexity,
            dist=dist,
            rho=rho,
            gamma=self._CalGamma(v_input),
            v=v_input,
            pow=self.args['pow_input'],
        )
        sigma = np.array(r.Getout())

        std_dis = np.std(rho) / np.sqrt(X.shape[1])
        print('std_dis', std_dis)
        if self.same_sigma is True:
            # sigma[:] = sigma.mean() * 5
            sigma[:] = sigma.mean()
            rho[:] = 0
        # print('sigma', sigma[:10])

        return rho, sigma

    def _initKNN(self, X, perplexity, v_input, K=500):

        print('use kNN mehtod to find the sigma')

        X_rshaped = X.reshape((X.shape[0], -1))
        index = NNDescent(X_rshaped, n_jobs=-1, metric=self.args['metric'])
        self.neighbors_index, neighbors_dist = index.query(X_rshaped, k=K)
        neighbors_dist = np.power(neighbors_dist, 2)
        rho = neighbors_dist[:, 1]

        r = PoolRunner(
            similarity_function_nunpy=self.SimilarityNPF,
            number_point=X.shape[0],
            perplexity=perplexity,
            dist=neighbors_dist,
            rho=rho,
            gamma=self._CalGamma(v_input),
            v=v_input,
            pow=self.args['pow_input'],
        )
        sigma = np.array(r.Getout())

        std_dis = np.std(rho) / np.sqrt(X.shape[1])
        print('std_dis', std_dis)
        if std_dis < 0.20 or self.args['same_sigma'] is True:
            sigma[:] = sigma.mean()
        return rho, sigma

    def _CalRho(self, dist):
        dist_copy = np.copy(dist)
        row, col = np.diag_indices_from(dist_copy)
        dist_copy[row, col] = 1e16
        rho = np.min(dist_copy, axis=1)
        return rho

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        # if index in self.data:
        # data_item = self.data[index]
        # rho_item = self.rho[index]
        # sigma_item = self.sigma[index]
        # label_item = self.label[index]
        # # print(self.args['K'])
        # data_item2 = self.aug_near_mix(index, k=self.args['K'])
        # labelstr_item = self.label_str[index]

        # if self.trains is not None:
        #     data_item = self.trains(data_item)

        return index  # data_item, data_item2, rho_item, sigma_item, label_item, index

    def aug_near_mix(self, index, k=10):
        r = np.random.randint(1, k)
        random_select_near_index = self.neighbors_index[index, r]
        random_select_near_data2 = self.data[random_select_near_index]
        random_rate = np.random.rand()
        return random_rate * random_select_near_data2 + (1 - random_rate) * self.data[index]
