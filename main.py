import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np
import random
from tqdm import trange

from utils.distribute import uniform_distribute, train_dg_split
from utils.sampling import iid, noniid
from utils.options import args_parser
from src.update import ModelUpdate
from src.nets import MLP, CNN_v1, CNN_v2
from src.strategy import FedAvg
from src.test import test_img

writer = SummaryWriter()

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)     
    torch.cuda.manual_seed_all(args.seed) 

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        dg = copy.deepcopy(dataset)
        dataset_train = copy.deepcopy(dataset)
        
        dg_idx, dataset_train_idx = train_dg_split(dataset, args)
        
        dg.data, dataset_train.data = dataset.data[dg_idx], dataset.data[dataset_train_idx]
        dg.targets, dataset_train.targets = dataset.targets[dg_idx], dataset.targets[dataset_train_idx]
        
        # sample users
        if args.sampling == 'iid':
            dict_users = iid(dataset_train, args.num_users)
        elif args.sampling == 'noniid':
            dict_users = noniid(dataset_train, args)
        else:
            exit('Error: unrecognized sampling')
    
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        
        dg = copy.deepcopy(dataset)
        dataset_train = copy.deepcopy(dataset)
    
        dg_idx, dataset_train_idx = train_dg_split(dataset, args)
        
        dg.targets.clear()
        dataset_train.targets.clear()

        
        dg.data, dataset_train.data = dataset.data[dg_idx], dataset.data[dataset_train_idx]
        
        for i in list(dg_idx):
            dg.targets.append(dataset[i][1])
        for i in list(dataset_train_idx):
            dataset_train.targets.append(dataset[i][1])

        # sample users
        if args.sampling == 'iid':
            dict_users = iid(dataset_train, args.num_users)
        elif args.sampling == 'noniid':
            dict_users = noniid(dataset_train, args)
        else:
            exit('Error: unrecognized sampling')
    
    else:
        exit('Error: unrecognized dataset')
    
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNN_v2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_v1(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
        
    # initialization stage of FedShare
    initialization_stage = ModelUpdate(args=args, dataset=dataset, idxs=set(dg_idx))
    w_glob, _ = initialization_stage.train(local_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
    net_glob.load_state_dict(w_glob)

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    
    # distribute globally shared data (uniform distribution)
    share_idx = uniform_distribute(dg, args)
    
    for iter in trange(args.rounds):
        
        if not args.all_clients:
            w_locals = []
        
        m = max(int(args.frac * args.num_users), 1)
        
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            
            # Local update
            local = ModelUpdate(args=args, dataset=dataset, idxs=set(list(dict_users[idx]) + share_idx))
            
            w, loss = local.train(local_net = copy.deepcopy(net_glob).to(args.device), net = copy.deepcopy(net_glob).to(args.device))
            
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))    
        
        # update global weights
        w_glob = FedAvg(w_locals, args)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        
        if args.debug:
            print(f"Round: {iter}")
            print(f"Test accuracy: {acc_test}")
            print(f"Test loss: {loss_test}")
        
        # tensorboard
        if args.tsboard:
            writer.add_scalar(f"Test accuracy:Share{args.dataset}, {args.fed}", acc_test, iter)
            writer.add_scalar(f"Test loss:Share{args.dataset}, {args.fed}", loss_test, iter)
                
    writer.close()