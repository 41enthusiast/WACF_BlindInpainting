import cv2
import os
from os import listdir
from os.path import isfile
import random
import re
import PIL
from PIL import Image
import random
import torch
import torchvision
from torchvision.transforms import ToTensor


# temporary code, dataset must be processed for validation too
def canny_edge(image, low_threshold=100, high_threshold=200):
    # If input is color, convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

class ArtPainting_processedDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, parse_patches=True, conditional_map_type = 'canny_edges'):
        super().__init__()

        artpainting_dir = dir
        input_names, gt_names, mask_names, gt_mask_names = [], [], [], []

        artpainting_inputs = os.path.join(artpainting_dir, 'masked')
        images = [f for f in listdir(artpainting_inputs) if isfile(os.path.join(artpainting_inputs, f))]
        # assert len(images) == 50000
        input_names += [os.path.join(artpainting_inputs, i) for i in images]
        gt_names += [os.path.join(os.path.join(artpainting_dir, 'image'), i) for i in images]
        mask_names += [os.path.join(os.path.join(artpainting_dir, 'conditional_maps', conditional_map_type, 'masked'), i) for i in images]
        gt_mask_names += [os.path.join(os.path.join(artpainting_dir, 'conditional_maps', conditional_map_type, 'image'), i) for i in images]
        print(len(input_names), 'images in dataset')

        x = list(enumerate(input_names))
        random.shuffle(x)
        indices, input_names = zip(*x)
        gt_names = [gt_names[idx] for idx in indices]
        self.dir = None

        self.input_names = input_names
        self.gt_names = gt_names
        self.mask_names = mask_names
        self.gt_mask_names = gt_mask_names

        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    @staticmethod
    def resize_by_short_side(image, target_short=512, resample=PIL.Image.Resampling.LANCZOS):
        w, h = image.size
        if min(w, h) < target_short:
            new_w = (w + 15) // 16 * 16
            new_h = (h + 15) // 16 * 16
        else:
            if w < h:
                new_w = target_short
                new_h = int(h * target_short / w)
            else:
                new_h = target_short
                new_w = int(w * target_short / h)
            new_w = (new_w + 15) // 16 * 16
            new_h = (new_h + 15) // 16 * 16

        return image.resize((new_w, new_h), resample=resample)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        mask_name = self.mask_names[index]
        gt_mask_name = self.gt_mask_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            mask_img = PIL.Image.open(os.path.join(self.dir, mask_name)) if self.dir else PIL.Image.open(mask_name)
        except:
            # temporary solution. Generate maps for the masked version alr - use unguided diffusion logic
            mask_img = Image.fromarray(canny_edge(np.array(input_img)))
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
            gt_mask_img = PIL.Image.open(os.path.join(self.dir, gt_mask_name)) if self.dir else PIL.Image.open(gt_mask_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')
            gt_mask_img = Image.fromarray(canny_edge(np.array(gt_img)))

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            mask_img = self.n_random_crops(mask_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            gt_mask_img = self.n_random_crops(gt_mask_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]

            # masked and gt image channel concatenated patches, image name, mask of images
            return torch.stack(outputs, dim=0), img_id, torch.stack([self.transforms(mask_img[i]) for i in range(self.n)], dim=0), torch.stack([self.transforms(gt_mask_img[i]) for i in range(self.n)], dim=0)
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            input_img = self.resize_by_short_side(input_img, 1024)
            gt_img = self.resize_by_short_side(gt_img, 1024)
            mask_img = self.resize_by_short_side(mask_img, 1024)
            gt_mask_img = self.resize_by_short_side(gt_mask_img, 1024)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id, self.transforms(mask_img), self.transforms(gt_mask_img)

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

class ArtPainting_processed:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='art_painting'):
        print("=> evaluating art painting test set...")
        train_dataset = ArtPainting_processedDataset(dir=os.path.join(self.config.data.data_dir, 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        parse_patches=parse_patches)
        val_dataset = ArtPainting_processedDataset(dir=os.path.join(self.config.data.data_dir, 'test'),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader
