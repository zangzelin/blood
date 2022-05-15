import main_zzl
import baseline_zzl
import numpy as np
import wandb
import sys
import os
import pytorch_lightning as pl

def warper(args):
    
    info = [str(s) for s in sys.argv[1:]]
    runname = '_'.join(['dmt', args.data_name, ''.join(info)])
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

    train_acc_list = []
    train_auc_list = []
    val_acc_list = []
    val_auc_list = []
    test_acc_list = []
    test_auc_list = []
    
    for i in range(10):
        args.foldindex = i
        if args.method == 'dmt':
            log_dict = main_zzl.main(args)
        else:
            log_dict = baseline_zzl.main(args)

        train_acc_list.append(log_dict['train_acc'])
        train_auc_list.append(log_dict['train_auc'])
        val_acc_list.append(log_dict['val_acc'])
        val_auc_list.append(log_dict['val_auc'])
        test_acc_list.append(log_dict['test_acc'])
        test_auc_list.append(log_dict['test_auc'])
    
    index = np.argmax(val_auc_list)
    best_train_acc = train_acc_list[index]
    best_train_auc = train_auc_list[index]
    best_val_acc = val_acc_list[index]
    best_val_auc = val_auc_list[index]
    best_test_acc = test_acc_list[index]
    best_test_auc = test_auc_list[index]

    wandb.log({
        'best_index': index,
        'train_acc_list': train_acc_list,
        'train_auc_list': train_auc_list,
        'val_acc_list': val_acc_list,
        'val_auc_list': val_auc_list,
        'test_acc_list': test_acc_list,
        'test_auc_list': test_auc_list,
        'best_train_acc': best_train_acc,
        'best_train_auc': best_train_auc,
        'best_val_acc': best_val_acc,
        'best_val_auc': best_val_auc,
        'best_test_acc': best_test_acc,
        'best_test_auc': best_test_auc,
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
    parser.add_argument('--method', type=str, default="dmt", choices=[
        'dmt',
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
    parser.add_argument('--p1_index', type=int, default=0)
    parser.add_argument('--p2_index', type=int, default=0)
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
    # print(args.foldindex)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" if args.foldindex < 5 else "1"
    warper(args)