
import cv2
import glob
import numpy as np
from skimage.io import imread
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image


def get_imagelist_dataloader(batch_size=30, dataset_object=None):
    """dataloader with (64, 64) images."""
    if not dataset_object:
        raise Exception('Must provide a Dataset object')

    imagelist_loader = DataLoader(dataset_object, batch_size=batch_size, shuffle=True)

    return imagelist_loader

class ImageListDataset(Dataset):
    """Dress Sleeve Attribute Images - 216 x 261 x 3 for the most part."""
    def __init__(self, list_of_image_paths, transform=None):
        self.img_paths = list_of_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        #sample = imread(sample_path)
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
    
class ImageListDataset(Dataset):
    """Dress Sleeve Attribute Images - 216 x 261 x 3 for the most part."""
    def __init__(self, list_of_image_paths, transform=None, cut_from=None, cut_amount=None):
        self.img_paths = list_of_image_paths
        self.transform = transform
        self.cut_from = cut_from
        self.cut_amount = cut_amount

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        #sample = imread(sample_path)
        sample = Image.open(sample_path)
        #print(np.array(sample).shape)

        if self.cut_from:
            sample = np.array(sample)
            
            if self.cut_from == 'top':
                sample = sample[:self.cut_amount]
            if self.cut_from == 'bottom':
                sample = sample[self.cut_amount:]
            
            sample = Image.fromarray(sample)
    
        if self.transform:
            sample = self.transform(sample)   

        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0