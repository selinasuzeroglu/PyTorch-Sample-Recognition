import pandas as pd
from torch.utils.data import Dataset
from torchvision import *
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
        img_class = int(self.csv_file.iloc[idx, 1])
        # img_dir + image_name should give path to each individual image, f.e. \\images\\example0.jpg
        img_path = os.path.join(self.img_dir, img_name)
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

        normalize = transforms.Normalize(
            mean=[img_r_mean, img_g_mean, img_b_mean],
            std=[img_r_std, img_g_std, img_b_std])

        image = normalize(image_tensor)

        return image_tensor, img_class

