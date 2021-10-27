import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F


#resize to 28*28
train_data =
test_data =

batch_size = 64

#Create data loaders
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#Set cpu for training
device = 'gpu'

print('Using {} to train'.format(device))

class NerualNetwork(nn.Module):
    def __init__(self):
        super(NerualNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 2500), #bias=True
            nn.ReLU(),
            nn.Linear(2500, 2000),  # bias=True
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#Send the job
model = NerualNetwork().to(device)
print(model)


#Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        #Send the batch and job to device
        X, y = X.to(device), y.to(device)

        #Assess the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        optimizer.zero_grad() #for every mini-batch during the training phase, we need to explicitly set the gradients to zero before starting to do backpropragation
        loss.backward()
        optimizer.step() #optimizer.step performs a parameter update based on the current gradient

        if batch % 100 == 0: #batch == 100*k
            loss, current = loss.item(), batch * len(X)
            print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')

#Check model's performance against test data to ensure it's learning
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    #Avg
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {100*correct}:>0.1f%, Avg loss: {test_loss:>8f} \n')


#Start the process
epochs = 5
for t in range(epochs):
    print(f'Epoch {t+1}\n-----------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    test(train_dataloader, model, loss_fn)

print("All Done. Thank you!")


#Save model
model_name = f'model_{time.localtime(time.time())}.pth'
torch.save(model.state_dict(), model_name)
print(f'Saved PyTorch Model State to {model_name}')