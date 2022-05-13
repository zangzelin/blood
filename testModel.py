import torch
import os
# import load_data_f.dataset as datasetfunc
from load_data_f.dataset import BlodNoMissingDataModuleTrain, BlodNoMissingDataModuleVali, BlodNoMissingDataModuleTest
import load_disF.disfunc as disfunc
import load_simF.simfunc as simfunc
import pytorch_lightning as pl
from main import DMT_Model
from torch import tensor
from torch.nn import MSELoss

def parse():
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
    parser.add_argument('--num_latent_dim', type=int, default=1)
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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--computer', type=str, default=os.popen('git config user.name').read()[:-1])

    args = pl.Trainer.add_argparse_args(parser)
    args = args.parse_args()

    return args


if __name__ == "__main__":
    args = parse()
    disfunc_use = getattr(disfunc, 'EuclideanDistanceNumpy')
    simfunc_use = getattr(simfunc, 'UMAPSimilarity')
    simfunc_npuse = getattr(simfunc, 'UMAPSimilarityNumpy')

    # train_dm_class = getattr(datasetfunc, 'BlodNoMissingDataModuleTrain')
    train_dataset = BlodNoMissingDataModuleTrain(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        **args.__dict__,
        )
    # vali_dm_class = getattr(datasetfunc, 'BlodNoMissingDataModuleVali')
    vali_dataset = BlodNoMissingDataModuleVali(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        **args.__dict__,
        )
    # test_dm_class = getattr(datasetfunc, 'BlodNoMissingDataModuleTest')
    test_dataset = BlodNoMissingDataModuleTest(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        SimilarityNPF=simfunc_npuse,
        **args.__dict__,
        )

    model = DMT_Model(
        DistanceF=disfunc_use,
        SimilarityF=simfunc_use,
        Traindataset=train_dataset,
        Validataset=vali_dataset,
        Testdataset=test_dataset,
        **args.__dict__,
        )
    # model.load_state_dict(torch.load('./checkpoints/result.pth'))
    # # print(vali_dataset.datatrain)
    model.cuda()
    pre = model(torch.tensor(vali_dataset.data).float())[1].detach().cpu().reshape(vali_dataset.data.shape[0],)
    # print(pre.shape)
    # print(tensor(vali_dataset.label).shape)

    loss = MSELoss()(pre, tensor(vali_dataset.label))
    print("mse loss: ", loss)
