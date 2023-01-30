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



# 0) Prepare Images: Load and normalize the train and test data

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        # csv_file with image name [0], and class label [1]
        self.csv_file = pd.read_csv(csv_file)
        # image folder directory
        self.img_dir = img_dir

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = str(self.csv_file.iloc[idx, 0])
        img_class = str(self.csv_file.iloc[idx, 1])
        # img_dir + image_name should give path to each individual image, f.e. \\images\\example0.jpg
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path)

        # transform image to tensor with torch.Size([3, 224, 224])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])

        image_tensor = transform(image)

        # separate image_tensor into R,G,B layers for subsequent normalization
        img_r = image_tensor[0, :, :]
        img_g = image_tensor[1, :, :]
        img_b = image_tensor[2, :, :]

        img_r_mean = torch.mean(img_r).item()
        img_g_mean = torch.mean(img_g).item()
        img_b_mean = torch.mean(img_b).item()

        img_r_std = torch.std(img_r).item()
        img_g_std = torch.std(img_g).item()
        img_b_std = torch.std(img_b).item()

        normalize = transforms.Normalize((img_r_mean, img_g_mean, img_b_mean), (img_r_std, img_g_std, img_b_std))

        # normalize = transforms.Normalize(
        #     mean=[img_r_mean, img_g_mean, img_b_mean],
        #     std=[img_r_std, img_g_std, img_b_std])

        image = normalize(image_tensor)

        return image_tensor, img_class


""" ImageDataset.__getitem__(idx=str) returns tuple([0], [1]) 
with [0] = torch.tensor for image, [1] = string for corresponding label """


img_dir = 'C:\\Users\\ssuz0008\\PycharmProjects\\UVVis_3.0'
csv = 'C:\\Users\\ssuz0008\\PycharmProjects\\PyTorch\\file.csv'

dataset = ImageDataset(csv, img_dir)  # dataset


train_dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
"""train_dataloader.Size([1, 3, 224, 224]) 
for batch_size = 1, in_channels = 3 (RGB), pixel_height = 224, pixel_width = 224"""
# test_dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [2, 1])

dataiter = iter(train_dataloader)  # creating iterator which represents dataset
images, labels = next(dataiter)   # returning successive items in the iterator, i.e. tuple([image], [label])


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
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i in [0, 2]:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')