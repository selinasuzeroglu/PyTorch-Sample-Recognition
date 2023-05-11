import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# 0) Prepare Images: Load and normalize the train and test data

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # csv_file with image name [0], and class label [1]
        self.csv_file = pd.read_csv(csv_file)
        # image folder directory
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = str(self.csv_file.iloc[idx, 0])
        label = torch.tensor(self.csv_file.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])
# transform image to tensor with torch.Size([3, 224, 224])
