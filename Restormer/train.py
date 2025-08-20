import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from PIL import Image
import pandas as pd
from dataset import ArtDataset
from model import Restormer
from utils import rgb_to_y, psnr, ssim

def parse_args():
    desc = 'Pytorch Implementation of \'Restormer: Efficient Transformer for High-Resolution Image Restoration\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--data_name', type=str, default='art_painting', choices=['art_painting', 'Art_Dataset'])
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[4, 6, 6, 8],
                        help='number of transformer blocks for each level')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='number of attention heads for each level')
    parser.add_argument('--channels', nargs='+', type=int, default=[48, 96, 192, 384],
                        help='number of channels for each level')
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=300000, help='iterations of training')
    parser.add_argument('--batch_size', nargs='+', type=int, default=[64, 40, 32, 16, 8, 8],
                        help='batch size of loading images for progressive learning')
    
    #### THIS IS THE GPU BOTTLENECK #####
    parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 80, 96, 128, 164, 192],
                        help='patch size of each image for progressive learning')
    
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--milestone', nargs='+', type=int, default=[92000, 156000, 204000, 240000, 276000],
                        help='when to change patch size and batch size')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    # model_file is None means training stage, else means testing stage
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')

    return init_args(parser.parse_args([]))

class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
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

def test_loop(net, data_loader, num_iter):
    net.eval()
    total_psnr, total_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)
        for rain, norain, name, h, w in test_bar:
            rain, norain = rain.cuda(), norain.cuda()
            out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255).byte()
            # computer the metrics with Y channel and double precision
            y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            total_psnr += current_psnr.item()
            total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(args.save_path, args.data_name, name[0])
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            Image.fromarray(out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()).save(save_path)
            test_bar.set_description('Test Iter: [{}/{}] PSNR: {:.2f} SSIM: {:.3f}'
                                     .format(num_iter, 1 if args.model_file else args.num_iter,
                                             total_psnr / count, total_ssim / count))
    return total_psnr / count, total_ssim / count

def save_loop(net, data_loader, num_iter):
    global best_psnr, best_ssim
    val_psnr, val_ssim = test_loop(net, data_loader, num_iter)
    results['PSNR'].append('{:.2f}'.format(val_psnr))
    results['SSIM'].append('{:.3f}'.format(val_ssim))
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args.model_file else num_iter // 1000) + 1))
    data_frame.to_csv('{}/{}.csv'.format(args.save_path, args.data_name), index_label='Iter', float_format='%.3f')
    if val_psnr > best_psnr and val_ssim > best_ssim:
        best_psnr, best_ssim = val_psnr, val_ssim
        with open('{}/{}.txt'.format(args.save_path, args.data_name), 'w') as f:
            f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr, best_ssim))
        torch.save(model.state_dict(), '{}/{}.pth'.format(args.save_path, args.data_name))
        
args = parse_args()
test_dataset = ArtDataset('../../data','art_painting','test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

results, best_psnr, best_ssim = {'PSNR': [], 'SSIM': []}, 0.0, 0.0
model = Restormer(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).to(device)
if args.model_file:
    model.load_state_dict(torch.load(args.model_file))
    save_loop(model, test_loader, 1)
else:
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
    total_loss, total_num, results['Loss'], i = 0.0, 0, [], 0
    train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True)
    for n_iter in train_bar:
        # progressive learning
        if n_iter == 1 or n_iter - 1 in args.milestone:
            end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
            start_iter = args.milestone[i - 1] if i > 0 else 0
            length = args.batch_size[i] * (end_iter - start_iter)
            # train_dataset = RainDataset(args.data_path, args.data_name, 'train', args.patch_size[i], length)
            train_dataset = ArtDataset('../../data','art_painting','train', args.patch_size[i], length)
            train_loader = iter(DataLoader(train_dataset, args.batch_size[i], True, num_workers=args.workers))
            i += 1
        # train
        model.train()
        rain, norain, name, h, w = next(train_loader)
        rain, norain = rain.to(device), norain.to(device)
        out = model(rain)
        loss = F.l1_loss(out, norain)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_num += rain.size(0)
        total_loss += loss.item() * rain.size(0)
        train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                  .format(n_iter, args.num_iter, total_loss / total_num))

        lr_scheduler.step()
        if n_iter % 1000 == 0:
            results['Loss'].append('{:.3f}'.format(total_loss / total_num))
            save_loop(model, test_loader, n_iter)