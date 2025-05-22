# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:28:00 2025

@author: menas
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 60

# 2. Data Preparation
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)  # compressed representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # pixel values in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 4. Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for data, _ in train_loader:
        img = data.view(data.size(0), -1).to(device)
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Saving the Model
torch.save(model.state_dict(), 'autoencoder.pth')

# 6. Visualizing Original and Reconstructed Images
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    # Get a batch of test images
    sample = next(iter(train_loader))[0][:8]  # take first 8 images
    sample_flat = sample.view(sample.size(0), -1).to(device)
    reconstructed = model(sample_flat).view(-1, 1, 28, 28).cpu()

# Plotting
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    # Original images
    axes[0, i].imshow(sample[i][0], cmap='gray')
    axes[0, i].axis('off')
    # Reconstructed images
    axes[1, i].imshow(reconstructed[i][0], cmap='gray')
    axes[1, i].axis('off')

plt.suptitle("Top row: Original | Bottom row: Reconstructed", fontsize=14)
plt.tight_layout()
plt.show()