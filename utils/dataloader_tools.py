
import glob
import numpy as np
from skimage.io import imread
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import torchvision.transforms.functional as TF 

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
            try:
                if len(np.array(sample).shape) != 3:
                    print("file {} does not have 3 channels".format(self.img_paths[idx]))
                    print("Replacing with previous image")
                    sample = Image.open(self.img_paths[idx-1])
                elif np.array(sample).shape[2] != 3:
                    print("file {} does not have 3 channels".format(self.img_paths[idx]))
                    print("Replacing with previous image")
                    sample = Image.open(self.img_paths[idx-1])
            except:
                print(self.img_paths[idx] + ",")
                print("Image not found, replacing with previous")
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
        self.center_crop = transforms.CenterCrop((128,128))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        #sample = imread(sample_path)
        sample = Image.open(sample_path) # Image.open() -> H, W, C
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

        center = self.center_crop(sample)

        if self.transform:
            sample = self.transform(sample) 

        # cutting into 5 pieces
        # npimg = np.array(sample)
        # print(npimg.shape)
        # tophalf, bottomhalf = np.split(npimg,2,axis=2)
        # topleft, topright = np.split(tophalf,2,axis=1)
        # bottomleft, bottomright = np.split(bottomhalf,2,axis=1)

        #print("sample shape before torch split: ",sample.shape)
        tophalf, bottomhalf = torch.split(sample,128,dim=1)
        #splittop = torch.split(tophalf, 2, dim=1)
        #print("tophalf split shape: ", splittop.shape)
        #print("half shapes: ", tophalf.shape, bottomhalf.shape, type(tophalf), type(bottomhalf))
        topleft, topright = torch.split(tophalf, 128, dim=2)
        bottomleft, bottomright = torch.split(bottomhalf, 128, dim=2)
        #print("sample shape: ",sample.shape, type(sample))
        #print("half shapes: ", tophalf.shape, bottomhalf.shape, type(tophalf), type(bottomhalf))
        #print("quarter shapes: ", topleft.shape, topright.shape, bottomleft.shape, bottomright.shape)
        center = TF.to_tensor(center)
        #print(type(sample),type(center),type(topleft),type(topright),type(bottomleft),type(bottomright))
        #print("quarter shapes: ", center.shape, topleft.shape, topright.shape, bottomleft.shape, bottomright.shape)
    
        #sample = Image.fromarray(sample)
        #center = Image.fromarray(center)
        # topleft = Image.fromarray(topleft)
        # topright = Image.fromarray(topright)
        # bottomleft = Image.fromarray(bottomleft)
        # bottomright = Image.fromarray(bottomright)

        #sample = Image.fromarray(sample)
        # center = TF.to_tensor(np.array(center))#.permute(2,0,1)
        # topleft = TF.to_tensor(topleft).permute(2,0,1)
        # topright = TF.to_tensor(topright).permute(2,0,1)
        # bottomleft = TF.to_tensor(bottomleft).permute(2,0,1)
        # bottomright = TF.to_tensor(bottomright).permute(2,0,1)

        # print("sample shape: ",sample.shape, type(sample))
        # print("half shapes: ", tophalf.shape, bottomhalf.shape, type(tophalf), type(bottomhalf))
        # print("quarter shapes: ", center.shape, topleft.shape, topright.shape, bottomleft.shape, bottomright.shape)

        # Since there are no labels, we just return 0 for the "label" here
        # Can pass lists or dicts 
        return [sample, topleft, topright, bottomleft, bottomright, center], 0