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
data = ImageDataset(csv, img_dir)


def show_image(idx):
    image = data.__getitem__(idx)
    csv_file = pd.read_csv(csv)
    label = str(csv_file.iloc[idx, 1])
    print(f"Label: {label}")
    plt.imshow(image.permute(1, 2, 0))
    plt.show()



