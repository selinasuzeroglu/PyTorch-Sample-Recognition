import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    transforms.CenterCrop(200)])
    #transforms.ToTensor()])
# transform image to tensor with torch.Size([3, 224, 224])



img_dir = 'C:\\Users\\ssuz0008\\PycharmProjects\\UVVis_3.0\\Main_Arduino\\Photos'
csv = 'C:\\Users\\ssuz0008\\PycharmProjects\\PyTorch\\file.csv'

batch_size = 1


dataset = ImageDataset(csv, img_dir, transform=transform)

def imshow(img):
    img = img
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()







# # separate image_tensor into R,G,B layers for subsequent normalization
# img_r = image_tensor[0, :, :]
# img_g = image_tensor[1, :, :]
# img_b = image_tensor[2, :, :]
#
# img_r_mean = torch.mean(img_r).item()
# img_g_mean = torch.mean(img_g).item()
# img_b_mean = torch.mean(img_b).item()
#
# img_r_std = torch.std(img_r).item()
# img_g_std = torch.std(img_g).item()
# img_b_std = torch.std(img_b).item()
#
# normalize = transforms.Normalize((img_r_mean, img_g_mean, img_b_mean), (img_r_std, img_g_std, img_b_std))
#
# # normalize = transforms.Normalize(
# #     mean=[img_r_mean, img_g_mean, img_b_mean],
# #     std=[img_r_std, img_g_std, img_b_std])
#
# image = normalize(image_tensor)