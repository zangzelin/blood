from matplotlib.pyplot import axis
import numpy as np
from math import floor
from copy import deepcopy
import torch 
from torch import nn, optim, tensor, Tensor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from baseline import traditionWay
from loaddata import preprocess
from utils import maketrainData


def useAE(args, fullData, fullLabel, missingData, missingLabel):
    class AE(nn.Module):
        def __init__(self):
            super(AE, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(18, 12),
                                        nn.Linear(12, 8),
                                        nn.Linear(8, 4))
            self.decoder = nn.Sequential(nn.Linear(4, 8),
                                        nn.Linear(8, 12),
                                        nn.Linear(12, 18))
        def forward(self, x) -> torch.Tensor:
            encode = self.encoder(x)
            decode = self.decoder(encode)
            return decode
    
    class loss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            x_local = x.copy()
            # x_local = x.cpu()
            mask = x_local < 0
            mask[:, :7] = False

            x_local[mask] = 0

            # print(mask[:, 7:9])
            print(np.max(x_local[:, 7:9].numpy()))
            print(np.min(x_local[:, 7:9].numpy()))
            return torch.mean((y - x) ** 2) * 0.5

    def train(ae, optimizer, criterion, rf, data: Tensor, label: Tensor, epoches=1000):
        rf.fit(data, label)
        
        input_model_data = maketrainData(data)
        mask = input_model_data == -1
        
        dev_full_data = input_model_data.cuda()

        for epoch in range(epoches):
            # if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            output = ae(dev_full_data).cpu()
            print("max: ", np.max(output[:, 7:9].detach().numpy()))
            print("min: ", np.min(output[:, 7:9].detach().numpy()))
            loss = criterion(data[mask], output[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.log:
                if (epoch + 1) % 100 == 0:
                    print("epoch: {}, loss is {}".format((epoch + 1), loss.data))

        # print(np.max(input_model_data[:, 7:].cpu().numpy(), axis=0))
        # print(np.max(output[:, 7:].detach().numpy(), axis=0))
        # print(np.min(output[:, 7:].detach().numpy(), axis=0))

    def test(ae, rf, data: Tensor, label: Tensor):
        dev_missing_data = data.cuda()
        
        filledData = ae(dev_missing_data).cpu().detach().numpy()
        result = rf.predict(filledData)
        result[result < 0] = 0
        score = mean_squared_error(result, label)

        # print("score: ", score)
        return score

    batchsize = args.batchsize
    lr = args.lr
    weight_decay = args.weight_decay
    epoches = args.epoches

    kf = KFold(n_splits=10)
    torch.cuda.empty_cache()
    model = AE()
    rf = RandomForestRegressor(n_estimators=50)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if torch.cuda.is_available():
        model = model.cuda()

        scores = []        
        train(model, optimizer, criterion, rf, fullData, fullLabel, epoches)
        scores.append(test(model, rf, missingData, missingLabel))
        # for train_index, _ in kf.split(fullData):
        #     # print(fullData[train_index], '\n', fullLabel[train_index])
        #     train(model, optimizer, criterion, rf, fullData[train_index], fullLabel[train_index], epoches)
        # for test_index, _ in kf.split(missingData):
        #     # print(missingData[test_index], '\n', missingLabel[test_index])
        #     scores.append(test(model, rf, missingData[test_index], missingLabel[test_index]))
        print("average score: {}".format(np.mean(scores)))

def useUMAP(data, label):
    import umap
    import matplotlib.pyplot as plt
    reducer = umap.UMAP()
    local_data, local_label = deepcopy(data), deepcopy(label)
    embedding = reducer.fit_transform(local_data)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, title='Decomposition using UMAP')
    # ax.scatter(embedding[:, 0], embedding[:, 1], c=local_label, s=3)
    for i in np.unique(label):
        ax.scatter(embedding[label == i, 0], embedding[label == i, 1], label=i, s=3)
    ax.legend(np.unique(local_label), loc='upper right')
    fig.savefig('umap.jpg')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('-n', '--knn_neighbors', type=int, default=2, )
    parser.add_argument('-g', '--log', type=bool, default=True, )
    parser.add_argument('-i', '--if_norm', type=bool, default=True, )
    parser.add_argument('-p', '--preSave', type=bool, default=True, )

    parser.add_argument('-b', '--batchsize', type=int, default=900, )
    parser.add_argument('-e', '--epoches', type=int, default=100, )
    parser.add_argument('-l', '--lr', type=float, default=1e-4, )
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-5, )
    # args = argparse.add_argparse_args(parser)
    args = parser.parse_args()

    (
        nNull, missing, fillWithMean, fillWithMiddle, fillWithKNN,
    ) = preprocess(knn_neighbors=args.knn_neighbors, ifNorm=args.if_norm, preSave=args.preSave)

    nNullData, nNullLabel = nNull['data'], nNull['label']
    missingData, missingLabel = missing['data'],missing['label']

    # fillWithMeanData, fillWithMeanLabel = fillWithMean['data'],missing['label']
    # fillWithMiddleData, fillWithMiddleLabel = fillWithMiddle['data'],missing['label']
    # fillWithKNNData, fillWithKNNLabel = fillWithKNN['data'],missing['label']
    # torch.save(model.state_dict(), 'log/output/AE.pth')

    # with open('log/result_{}.log'.format(args.knn_neighbors), 'w') as f:
    # traditionWay({'random_forest': (RandomForestRegressor(n_estimators=50), None)}, nNullData, nNullLabel)
    useAE(args, tensor(nNullData).float(), tensor(nNullLabel).float(), tensor(missingData).float(), tensor(missingLabel).float())
    # print(nNullData, '\n', nNullLabel, '\n', missingData, '\n', missingLabel)

    # nNullLabel[nNullLabel > 0.5] = 3
    # nNullLabel[(nNullLabel > 0.3) & (nNullLabel <= 0.5)] = 2
    # nNullLabel[(nNullLabel > 0) & (nNullLabel <= 0.3)] = 1
    # nNullLabel[nNullLabel == 0] = 0

    # print(len(nNullLabel))
    # print(len(nNullLabel[nNullLabel == 0]))
    # print(len(nNullLabel[nNullLabel == 1]))
    # print(len(nNullLabel[nNullLabel == 2]))
    # print(len(nNullLabel[nNullLabel == 3]))
    # useUMAP(nNullData, nNullLabel)
    
    # ae_train_data = maketrainData(nNullData)
    
    # mask = ae_train_data == -1

    # print(ae_train_data[mask])
