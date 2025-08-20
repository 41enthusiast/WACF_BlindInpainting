from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from PIL import Image
from utils import pad_image_needed
import glob
import torchvision.transforms.functional as T
import os
import torch
import torch.nn.functional as F

class ArtDataset(Dataset):
    def __init__(self, data_path, data_name, data_type, patch_size=None, length=None, masked_fol = 'masked', clean_fol = 'image'):
        super().__init__()
        self.data_name, self.data_type, self.patch_size = data_name, data_type, patch_size
        self.masked_images = sorted(glob.glob('{}/{}/{}/{}/*.png'.format(data_path, data_name, data_type, masked_fol)))
        self.clean_images = sorted(glob.glob('{}/{}/{}/{}/*.png'.format(data_path, data_name, data_type, clean_fol)))
        # make sure the length of training and testing different
        self.num = len(self.masked_images)
        self.sample_num = length if data_type == 'train' else self.num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.masked_images[idx % self.num])
        masked = T.to_tensor(Image.open(self.masked_images[idx % self.num]))
        clean = T.to_tensor(Image.open(self.clean_images[idx % self.num]))
        h, w = masked.shape[1:]

        if self.data_type == 'train':
            # make sure the image could be cropped
            masked = pad_image_needed(masked, (self.patch_size, self.patch_size))
            clean = pad_image_needed(clean, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(masked, (self.patch_size, self.patch_size))
            masked = T.crop(masked, i, j, th, tw)
            clean = T.crop(clean, i, j, th, tw)
            if torch.rand(1) < 0.5:
                masked = T.hflip(masked)
                clean = T.hflip(clean)
            if torch.rand(1) < 0.5:
                masked = T.vflip(masked)
                clean = T.vflip(clean)
        else:
            # padding in case images are not multiples of 8
            new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8# +8 to account for minimum h n w <8
            pad_h = new_h - h if h % 8 != 0 else 0
            pad_w = new_w - w if w % 8 != 0 else 0
            masked = F.pad(masked, (0, pad_w, 0, pad_h), 'reflect')
            clean = F.pad(clean, (0, pad_w, 0, pad_h), 'reflect')
        return masked, clean, image_name, h, w