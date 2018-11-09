
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
    def __init__(self, list_of_image_paths, 
                transform=None, cut_from=None, cut_amount=None, 
                convert_rgb=False, error_handling=False):
        self.img_paths = list_of_image_paths
        self.transform = transform
        self.cut_from = cut_from
        self.cut_amount = cut_amount
        self.convert_rgb = convert_rgb
        self.error_handling = error_handling

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        #sample = imread(sample_path)
        sample = Image.open(sample_path)
        #print(np.array(sample).shape)

        if self.error_handling:
            if len(np.array(sample).shape) != 3:
                print("file {} does not have 3 channels".format(self.img_paths[idx]))
                print("Replacing with previous image")
                sample = Image.open(self.img_paths[idx-1])
            elif np.array(sample).shape[2] != 3:
                print("file {} does not have 3 channels".format(self.img_paths[idx]))
                print("Replacing with previous image")
                sample = Image.open(self.img_paths[idx-1])

        if self.convert_rgb:
            sample = sample.convert("RGB")

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

class ImgDsetCut5from256(Dataset):
    """Assumes 256 x 256 x 3 image and cuts in 4 and adds addtl center crop"""
    def __init__(self, list_of_image_paths, 
                transform=None, convert_rgb=False, error_handling=False):
        self.img_paths = list_of_image_paths
        self.transform = transform
        self.convert_rgb = convert_rgb
        self.error_handling = error_handling
        self.center_crop = transforms.CenterCrop((64,64))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        #sample = imread(sample_path)
        sample = Image.open(sample_path)
        #print(np.array(sample).shape)

        if self.error_handling:
            if len(np.array(sample).shape) != 3:
                print("file {} does not have 3 channels".format(self.img_paths[idx]))
                print("Replacing with previous image")
                sample = Image.open(self.img_paths[idx-1])
            elif np.array(sample).shape[2] != 3:
                print("file {} does not have 3 channels".format(self.img_paths[idx]))
                print("Replacing with previous image")
                sample = Image.open(self.img_paths[idx-1])

        if self.convert_rgb:
            sample = sample.convert("RGB")

        if self.transform:
            sample = self.transform(sample) 

        # cutting into 5 pieces
        npimg = np.array(sample)
        tophalf, bottomhalf = np.split(npimg,2)
        topleft, topright = np.split(tophalf,128,axis=1)
        bottomleft, bottomright = np.split(bottomhalf,128,axis=1)
        center = self.center_crop(sample)

        print(tophalf.shape, bottomhalf.shape, topleft.shape, topright.shape)
            
        #sample = Image.fromarray(sample)
        topleft = Image.fromarray(topleft)
        topright = Image.fromarray(topright)
        bottomleft = Image.fromarray(bottomleft)
        bottomright = Image.fromarray(bottomright)

        # Since there are no labels, we just return 0 for the "label" here
        return [sample, topleft, topright, bottomleft, bottomright, center], 0