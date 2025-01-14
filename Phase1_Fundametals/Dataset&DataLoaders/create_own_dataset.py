import os 
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class customDataset(Dataset):
    def __init__(self,annotation_file, img_dir, transform = None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        img = read_image(img_path)
        label = self.img_labels.iloc[index,1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label    = self.target_transform(label)
        return img, label