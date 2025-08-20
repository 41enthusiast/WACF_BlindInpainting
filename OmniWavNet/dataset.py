from utils import pad_image_needed
import torch
import torchvision.transforms.functional as T
from torchvision.transforms import RandomCrop
import glob
import os
from PIL import Image
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, data_path, data_path_test, data_name, data_type, patch_size=None, length=None):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size

        self.corrupt_images = sorted(glob.glob('{}/masked/*.png'.format(data_path)))
        self.clear_images = sorted(glob.glob('{}/image/*.png'.format(data_path)))

        self.corrupt_images_test = sorted(glob.glob('{}/masked/*.png'.format(data_path_test)))
        self.clear_images_test = sorted(glob.glob('{}/image/*.png'.format(data_path_test)))

        self.num = len(self.corrupt_images)
        self.num_test = len(self.corrupt_images_test)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        if self.data_type == 'train':
            corrupt_image_name = os.path.basename(self.corrupt_images[idx % self.num])
            gt_image_name = os.path.basename(self.clear_images[idx % self.num])

            corrupt = T.to_tensor(Image.open(self.corrupt_images[idx % self.num]))
            clear = T.to_tensor(Image.open(self.clear_images[idx % self.num]))
            h, w = corrupt.shape[1:]
            # make sure the image could be cropped
            corrupt = pad_image_needed(corrupt, (self.patch_size, self.patch_size))
            clear = pad_image_needed(clear, (self.patch_size, self.patch_size))

            i, j, th, tw = RandomCrop.get_params(corrupt, (self.patch_size, self.patch_size))

            corrupt = T.crop(corrupt, i, j, th, tw)
            clear = T.crop(clear, i, j, th, tw)

            if torch.rand(1) < 0.5:
                corrupt = T.hflip(corrupt)
                clear = T.hflip(clear)
            if torch.rand(1) < 0.5:
                corrupt = T.vflip(corrupt)
                clear = T.vflip(clear)
        else:
            corrupt_image_name = os.path.basename(self.corrupt_images_test[idx % self.num_test])
            gt_image_name = os.path.basename(self.clear_images_test[idx % self.num_test])

            corrupt = T.to_tensor(Image.open(self.corrupt_images_test[idx % self.num_test]))
            clear = T.to_tensor(Image.open(self.clear_images_test[idx % self.num_test]))

            h, w = corrupt.shape[1:]
        return corrupt, clear, corrupt_image_name, h, w
