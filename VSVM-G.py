import argparse
import time

import numpy as np
import random
import pickle, os, cv2
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args(args=[])

device = torch.device("cuda")
torch.manual_seed(args.seed)
kwargs = {'num_workers': 0, 'pin_memory': True}


# Preparing Data
# dir = "F:/Surf Road Detect/TRAIN&TEST/trainImages"
#
# # categories = ['Pothole', 'Road Marking']
#
# categories = ['Alligator', 'Joint', 'Longitudal', 'Manholes', 'Oil Marks', 'Pothole', 'Road Marking', 'Shadow',
#               'Transverse']
#
# data = []
#
#
# # read the pickle file: pick_in
# pick_in = open('data1.pickle', 'rb')
# data = pickle.load(pick_in)
# pick_in.close()
#
# print('data read complete')
#
# random.shuffle(data)
#
# print('shuffle complete')
#
# features = []
# labels = []
#
# for feature, label in data:
#     features.append(feature)
#     labels.append(label)
#
# print('feature and label load complete')
#
#
# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
# print('split complete')


train_set = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)

test_set = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)




class SVM(nn.Module):
    """Support Vector Machine"""
    # SGD can't read model para issue
    def __init__(self):
        super(SVM, self).__init__()
        self.w = nn.Parameter(torch.randn(1, 784), requires_grad=True).cuda()
        self.b = torch.ones(128, 1).cuda()  # vsv

    # vsv
    def forward(self, x):
        print(x.size())
        print(((1 / 512)*torch.ones(1, 28)).size())
        print(((x.matmul(self.w.t()))).size())
        h = ((1 / 512)*torch.ones(1, 28)) @ ((x.matmul(self.w.t()) + self.b) ** 9)
        return h


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
    model = SVM().to(device)
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
        print('test_loss: {:.4f}, adv_acc: {:.2f}%'.format(testloss, 100. * testacc))


    return model

model = train_model()