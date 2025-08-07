import os
import random
import argparse
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as T
from torch.backends import cudnn

################################################################################################################
# Configurations
################################################################################################################

class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_path_test = args.data_path_test
        self.data_name = args.data_name
        self.save_path = args.save_path
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.finetune = args.finetune

def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    return Config(args)

def parse_args():
    desc = 'Pytorch Implementation of \'Restormer: Efficient Transformer for High-Resolution Image Restoration\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='/train/')
    parser.add_argument('--data_path_test', type=str, default='/test/')
    parser.add_argument('--data_name', type=str, default='inpaint', choices=['rain100L', 'rain100H', 'inpaint'])
    parser.add_argument('--save_path', type=str, default='test_places')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8],
                        help='number of transformer blocks for each level')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='number of attention heads for each level')
    parser.add_argument('--channels', nargs='+', type=int, default=[48//3, 96//3, 192//3, 384//3],
                        help='number of channels for each level')
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=9057800, help='iterations of training')
    parser.add_argument('--batch_size', nargs='+', type=int, default=[2, 2, 2, 2, 2, 2],
                        help='batch size of loading images for progressive learning')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[128, 160, 192, 256, 320, 384],
                        help='patch size of each image for progressive learning')
    parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate')
    parser.add_argument('--milestone', nargs='+', type=int, default=[1125000, 1800000, 840000, 2250000],
                        help='when to change patch size and batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    # model_file is None means training stage, else means testing stage
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')
    parser.add_argument('--finetune', default=True, help='path of pre-trained model file')

    return init_args(parser.parse_args([]))

################################################################################################################
# Data Transformations
################################################################################################################

def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)

def pad_image_needed(img, size):
    width, height = T.get_image_size(img)
    if width < size[1]:
        img = T.pad(img, [size[1] - width, 0], padding_mode='reflect')
    if height < size[0]:
        img = T.pad(img, [0, size[0] - height], padding_mode='reflect')
    return img

################################################################################################################
# Metrics
################################################################################################################


def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()

################################################################################################################
# Optimization
################################################################################################################

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
