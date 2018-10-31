
import cv2
import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transforms.Resize(image, (new_h, new_w),  interpolation=Image.BICUBIC,)

        return {'image': img}

class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0

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