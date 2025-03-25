import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Neural network model definition
class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super(NeuralNetworkModel, self).__init__()
        self.layer0_dense_0 = nn.Linear(in_features=784, out_features=128)
        self.layer1_dropout_0 = nn.Dropout(p=0.3)
        self.layer2_output_0 = nn.Sequential(nn.Linear(in_features=128, out_features=10), nn.Softmax(dim=1))

    # Forward pass
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.layer0_dense_0(x)
        x = self.layer1_dropout_0(x)
        x = self.layer2_output_0(x)
        return x

# Model instantiation
model = NeuralNetworkModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Loss function
loss_fn = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.004518206413675764)

# Mixed precision training setup
scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(data)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
