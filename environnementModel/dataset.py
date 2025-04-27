from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms


class DailyPlacesDataset(Dataset):
    def __init__(self, root_dir, transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.image_paths = []
        self.labels = []
        for label_id, label_name in enumerate(self.classes):
            label_dir = os.path.join(root_dir, label_name, "img")
            for image_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, image_name))
                self.labels.append(label_id)
                

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.labels[index]
    
    def getClasses(self):
        return self.classes 
