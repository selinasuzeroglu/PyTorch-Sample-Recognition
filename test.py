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
from Custom_Dataset import ImageDataset, transform

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


""" ImageDataset.__getitem__(idx=str) returns tuple([0], [1]) 
with [0] = torch.tensor for image, [1] = string for corresponding label """

img_dir = 'C:\\Users\\ssuz0008\\PycharmProjects\\UVVis_3.0\\Main_Arduino\\Photos'
csv = 'C:\\Users\\ssuz0008\\PycharmProjects\\PyTorch\\file.csv'

batch_size = 4

dataset = ImageDataset(csv, img_dir, transform=transform)

train_size = int(len(dataset)*.7)
test_size = int(len(dataset)-train_size)

train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
