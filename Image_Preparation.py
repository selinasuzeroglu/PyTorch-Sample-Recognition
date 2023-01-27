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
        self.csv_file = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = str(self.csv_file.iloc[idx, 0])
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path)
        img_class = str(self.csv_file.iloc[idx, 1])

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
        return image


img_dir = 'C:\\Users\\ssuz0008\\PycharmProjects\\UVVis_3.0\\Main_Arduino'
csv = 'C:\\Users\\ssuz0008\\PycharmProjects\\PyTorch\\file.csv'
dataset = ImageDataset(csv, img_dir)  # dataset


def show_image(idx):
    image = dataset.__getitem__(idx)
    csv_file = pd.read_csv(csv)
    label = str(csv_file.iloc[idx, 1])
    print(f"Label: {label}")
    plt.imshow(image.permute(1, 2, 0))
    plt.show()


batch_size = 1
num_workers = 1

# train_dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [3, 1])


class Net(nn.Module):
    ''' Models a simple Convolutional Neural Network'''

    def __init__(self):
        ''' initialize the network '''
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels,
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)