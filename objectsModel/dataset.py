import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class Calib_dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = [f'{self.root_dir}\{img_p}' for img_p in os.listdir(self.root_dir)]



    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
        img_test = Image.open(self.image_paths[index]).convert('RGB').resize((1920, 1080))

        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

        image = transform(img_test).unsqueeze(0)
        image = image.type(torch.float32)

        
        return image 
