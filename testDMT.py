# from scanpy.tools import dmt
# from scanpy.preprocessing import pca
# from scanpy.readwrite import read
# from scanpy.plotting import dmtp
import DMT
from DMT.tools import dmt
import numpy as np
# from DMT.readwrite import read
# from DMT.plotting import dmtp
from anndata import AnnData
import torch
from torch import tensor, cuda
# import shutil
import logging


# 方法支持重写，只需要返回data和label的tensor即可,dmt的初始化参数中需要sadata，data，label
def getData(sadata: AnnData):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    # import pdb
    # pdb.set_trace()
    sadata_pca = pca.fit_transform(sadata.obsm['X_pca'])
    sadata_pca = tensor(sadata_pca.copy())
    data_train = sadata_pca

    label_train_str = list(sadata.obs['celltype'])
    label_train_str_set = list(set(label_train_str))
    label_train = tensor([label_train_str_set.index(i) for i in label_train_str])

    return {"data": data_train.to(device), "label": label_train.to(device)}


def getSwissRoll(n_samples):
    from sklearn.datasets import make_swiss_roll
    data = make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=1)
    X = data[0]
    y = data[1]

    X[:, 0] = X[:, 0] - np.mean(X[:, 0])
    X[:, 1] = X[:, 1] - np.mean(X[:, 1])
    X[:, 2] = X[:, 2] - np.mean(X[:, 2])

    scale = 15 / max(
        np.max(X[:, 0]) - np.min(X[:, 0]),
        np.max(X[:, 1]) - np.min(X[:, 1]),
        np.max(X[:, 2]) - np.min(X[:, 2]),
    )
    X = X * scale
    # nP = X.shape[0] // 2
    # nP = X.shape[0]
    return {"data": tensor(X), "label": tensor(y)}


if __name__ == "__main__":
    # 没有GPU就使用CPU，使用GPU会快很多
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda' if cuda.is_available() else 'cpu')
    # 读取数据J
    # sadata = read('/usr/commondata/public/scRNAseqDataset/PBMC3k_HVG_regscale.h5ad')
    # print(getData(sadata)['data'].shape, getData(sadata)['label'].shape)

    # 初始化DMT
    # 如果需要打印或者保存日志，只需在此设置即可
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        # filename=os.path.join('20210621.log')
    )
    # data, label = getSwissRoll(40000).values()
    # torch.save(data, './data_2000.pth')
    # torch.save(label, './label_2000.pth')
#     data = torch.load('./data_2000.pth')
#     label = torch.load('./label_2000.pth')
    data = torch.load('data.pth')
    label = torch.load('label.pth')

    d = dmt(data=data, label=label, data_name="blood", device=device, lr=1e-3,
            batch_size=16000, epochs=1000, log_interval=200,
            # perplexity=10, version=1, NetworkStructure=[-1, 500, 500, 2], lr_iter=[500],)
            perplexity=5, version=2, NetworkStructure=[-1, 600, 500, 400, 300, 200, 2], lr_iter=[200], fb=1, nb=0, DEC=False)

    # 使用DMT训练并降维可视化，isSaveData设置是否保存可视化结果
    d.fit_transform(interData=True)
    # d.transform(data, label, preTrain='./2021061601_dmt.pth')

    # 使用dmtp画图，save可以为False，也可以直接设置成保存的文件名
    # dmtp(d.sadata, color='celltype', legend_loc="on data", legend_fontsize='xx-small', save=False)['perplexity'], d.args['vs']))
    # except Exception as e:
    #     print(e)
    #     shutil.rmtree('./' + d.path)
    # from numba import cuda
    # cuda.select_device(1)
    # print(cuda.get_current_device())
