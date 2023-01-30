import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        normalize = transforms.Normalize(
            mean=[img_r_mean, img_g_mean, img_b_mean],
            std=[img_r_std, img_g_std, img_b_std])

        image = normalize(image_tensor)

        # ImageDataset.__getitem__() returns tuple([0], [1]) with [0] = image, [1] = class label
        return image_tensor, img_class


img_dir = 'C:\\Users\\ssuz0008\\PycharmProjects\\UVVis_3.0'
csv = 'C:\\Users\\ssuz0008\\PycharmProjects\\PyTorch\\file.csv'
dataset = ImageDataset(csv, img_dir)  # dataset


def show_image(idx):
    image = dataset.__getitem__(idx)
    csv_file = pd.read_csv(csv)
    label = str(csv_file.iloc[idx, 1])
    print(f"Label: {label}")
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


# train_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [2, 1])

tuple = dataset.__getitem__(0)

print(type(tuple[1]))



# 2) Define Neural Network
#
# class Net(nn.Module):
#     ''' Models a simple Convolutional Neural Network'''
#
#     def __init__(self):
#         ''' initialize the network '''
#         super(Net, self).__init__()
#         # 3 input image channel, 6 output channels,
#         # 5x5 square convolution kernel
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         # Max pooling over a (2, 2) window
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         ''' the forward propagation algorithm '''
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# net = Net()
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

