import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.nn.utils import clip_grad_norm_

import numpy as np
from random import randint

from src.test import test_img


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ModelUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, local_net, net):
        
        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []

        if self.args.sys_homo: 
            local_ep = self.args.local_ep
        else:
            local_ep = randint(self.args.min_le, self.args.max_le)

        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # FedProx: https://arxiv.org/abs/1812.06127
                if self.args.fed == 'fedprox':
                    if iter > 0: 
                        w_diff = torch.tensor(0., device=self.args.device)
                        for w, w_t in zip(local_net.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        loss += self.args.mu / 2. * w_diff
                        w_t += self.args.mu * w_diff
                        
                loss.backward()
                
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)