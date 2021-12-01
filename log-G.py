############################################################################
### Written by Gaojie Jin and updated by Xiaowei Huang, 2021
###
### For a 2-nd year undergraduate student competition on
### the robustness of deep neural networks, where a student
### needs to develop
### 1. an attack algorithm, and
### 2. an adversarial training algorithm
###
### The score is based on both algorithms.
############################################################################
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time

# input id
id_ = 1000

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args(args=[])


device = torch.device("cuda")
torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True}


train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)


    def forward(self, x):
        output = self.linear(x)
        return output



'train function, you can use adversarial training'


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)

        # use adverserial data to train the defense model
        # adv_data = adv_attack(model, data, target, device=device)

        # clear gradients
        optimizer.zero_grad()

        # compute loss
        # loss = F.cross_entropy(model(adv_data), target)
        loss = F.cross_entropy(model(data), target)

        # get gradients and update
        loss.backward()
        optimizer.step()


'predict function'


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def train_model():
    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # training
        train(args, model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        testloss, testacc = eval_test(model, device, test_loader)

        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(testloss, 100. * testacc), end='\n')



    # save the model
    torch.save(model.state_dict(), str(id_) + '.pt')
    return model



'Comment out the following command when you do not want to re-train the model'
'In that case, it will load a pre-trained model you saved in train_model()'
if __name__ == '__main__':
    model = train_model()


