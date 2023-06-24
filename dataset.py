from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import numpy as np
from config import tr_config


class ImageNetDatasetForMoCoPretraining(Dataset):
    def __init__(self, data_path="/sdc1/caiyunuo/pretraining-ResNet/dataset/train", transform=None):
        super().__init__()
        self.image_list = glob(data_path+"/*/*.JPEG")
        if transform is not None:
            self.transform = transform
        else:
            self.transform = tr_config.augmentation

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(img_path).convert('RGB')
        q = self.transform(img)
        k = self.transform(img)
        return [q, k]
    
    def __len__(self):
        return len(self.image_list)


