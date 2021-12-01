import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import elasticdeform

# Set gpu for training
device = 'cuda'
torch.manual_seed(0)
kwargs = {'num_workers': 0, 'pin_memory': True}


# Create data loaders
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                              transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)

test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                             transform=transforms.Compose([ToTensor()]))
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)

lr = 1e-2
epochs = 50


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # elastic conv kernel
        self.fc1 = nn.Linear(28*28, 2500)
        self.fc2 = nn.Linear(2500, 2000)
        self.fc3 = nn.Linear(2000, 1500)
        self.fc4 = nn.Linear(1500, 1000)
        self.fc5 = nn.Linear(1000, 500)
        self.fc6 = nn.Linear(500, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, dataloader, optimizer, epoch):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        # Send the batch and job to device
        x, y = x.to(device), y.to(device)
        # x = elasticdeform.deform_random_grid(x, sigma=25)
        # y = elasticdeform.deform_random_grid(y, sigma=25)
        data = x.view(x.size(0), 28*28)

        # Backpropagation
        optimizer.zero_grad()

        loss = F.cross_entropy(model(x), y)

        loss.backward()
        optimizer.step()


# Check model's performance against test data to ensure it's learning
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28*28)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def model_train():
    # Send the model to gpu
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # training
        train(model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trainloss, trainacc = eval_test(model, device, train_loader)
        testloss, testacc = eval_test(model, device, test_loader)
        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('train_loss: {:.4f}, train_acc: {:.2f}%'.format(trainloss, 100. * trainacc), end=', ')
        print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(testloss, 100. * testacc), end='\n')


# Start the process
if __name__ == '__main__':
    model_train()

# #Save model
# model_name = f'model_{time.localtime(time.time())}.pth'
# torch.save(model.state_dict(), model_name)
# print(f'Saved PyTorch Model State to {model_name}')
