import math

import torch
from dataset import load_data
from torch.utils.data import DataLoader
from torch import nn
from model import MLP
from optimizer import SGD,ADAM,AdaGrad,RMSprop
from train import train
import pandas as pd
import argparse
#TODO 1: FW
# hyperparameters for fw
# 1. step size: algorithm to adjust(PFW: https://arxiv.org/pdf/1806.05123.pdf)
# *2. kappa: find rule      ws
# 3. fw for other optimizers
# *4. number of maximum gradients for lmo: is it ok to increase theoretically?
# 5. dataset: try different dataset and data size  jw cifar cnn
# *6. different networks, number of nn parameters  ws cnn
# 7. batch size
#TODO 2: projected mxq
# hyperparameters for projected XX
# 1. learning rate
# 2. other parameters for each optimizer, momentum, beta, etc.
# 3. batch size
#TODO 3:
# *lp norm? wj
# other constraint algorithm?
# plot: loss wrt time vs loss wrt epoch for fw vs projected jw
# plot: show l1 sparsity effect: avoid overfitting

torch.manual_seed(100)
batch_size = 1000
train_input, train_target, test_input, test_target = load_data(train_size=60000, test_size=10000, cifar = None,  normalize = False, flatten = True)
train_loader = DataLoader(list(zip(train_input, train_target)), batch_size)
test_loader = DataLoader(list(zip(test_input, test_target)), batch_size)
criterion = nn.CrossEntropyLoss()


kappas = [175, 5, 0.1]
ls = [1,2,math.inf]
for i in range(3):
    torch.manual_seed(100)
    model = MLP(input_channels = 14*14, output_channels = 10)
    optimizer = SGD(model.parameters(),  kappa = kappas[i], l = ls[i], step_size=1, project = False, FW = True, lr = 0.001, momentum = 0)
    loss_tr_hist, loss_te_hist, acc_train_hist, acc_test_hist = train(train_loader, test_loader,
              model, optimizer, criterion, epochs=100)




