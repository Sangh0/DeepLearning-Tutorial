import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Set hyperparameters
Config = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'epochs': 20,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load MNIST dataset
train_set = dsets.MNIST(
    root='mnist/',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_set = dsets.MNIST(
    root='mnist/',
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=Config['batch_size'],
    shuffle=True,
    drop_last=True,
)

test_loader = DataLoader(
    dataset=test_set,
    batch_size=Config['batch_size'],
    shuffle=True,
    drop_last=True,
)


# Build Model (Multi Layer Perceptron)
class DNN(nn.Module):

    def __init__(self, in_dim=768, hidden_dim=1000, out_dim=10):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.model(x)
        return out


# Compile
model = DNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Config['learning_rate'])

def cal_accuracy(outputs, labels):
    outputs = torch.argmax(outputs, dim=1)
    correct = (outputs, labels).sum()/len(outputs)
    return correct


# Training
for epoch in range(Config['epochs']):
    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        acc = cal_accuracy(outputs, labels)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        if (batch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{Config['epochs']}, Batch {batch+1}/{len(train_loader)}\n'
                  f'loss: {loss.item():.3f}, accuracy: {acc.item():.3f}')


# Testing
test_loss, test_acc = 0, 0
with torch.no_grad():
    model.eval()
    for batch, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        acc = cal_accuracy(outputs, labels)
        test_acc += acc.item()
        loss = loss_func(outputs, labels)
        test_loss += loss.item()

print(f'Test Loss: {test_loss/(batch+1):.3f}, Test Accuracy: {test_acc/(batch+1):.3f}')