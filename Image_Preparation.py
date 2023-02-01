import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import *
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils

import matplotlib

import numpy as np
from PIL import Image
from Custom_Dataset import ImageDataset


""" ImageDataset.__getitem__(idx=str) returns tuple([0], [1]) 
with [0] = torch.tensor for image, [1] = string for corresponding label """

# direction of training images:
train_img_dir = 'C:\\Users\\ssuz0008\\PycharmProjects\\UVVis_3.0\\Main_Arduino\\Photos'
# direction of training csv.file:
train_csv = 'C:\\Users\\ssuz0008\\PycharmProjects\\PyTorch\\file.csv'


train_dataset = ImageDataset(train_csv, train_img_dir)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
"""train_dataloader has elements consisting of tuple(image_tensor, labels) with image_tensor.Size([1, 3, 224, 224]) 
for batch_size = 1, in_channels = 3 (RGB), pixel_height = 224, pixel_width = 224
and labels.Size([1]) for integers representing class label"""

dataiter = iter(train_dataloader)  # creating iterator which represents dataset
images, labels = next(dataiter)  # returning successive items in the iterator, i.e. tuple([image], [label])

test_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False)

# showing images:
def imshow(img):
  ''' function to show image '''
  img = img
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


# 2) Define Neural Network

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size = width x height)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.adapt = nn.AdaptiveAvgPool2d(5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adapt(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 3) Define Loss Function

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 4) Train Network

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        # running_loss = 0.0

PATH = 'C:\\Users\\ssuz0008\\PycharmProjects\\PyTorch\\image_net.pth'
torch.save(net.state_dict(), PATH)

classes = (0, 1, 2)
# 5) Test Network

dataiter = iter(test_dataloader)
images, labels = next(dataiter)



outputs = net(images)

_, predicted = torch.max(outputs, 1)
print(' '.join('%s' % classes[predicted[j]] for j in range(1)))
imshow(torchvision.utils.make_grid(images))
