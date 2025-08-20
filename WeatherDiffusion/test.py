import torch
import numpy as np
import yaml
from utils import parse_args_and_config
from model import DenoisingDiffusion, DiffusiveRestoration
from dataset.artpainting_processed import ArtPainting_processed


resume, test_set, sampling_timesteps, grid_r, seed, image_folder, config = parse_args_and_config()

# setup device to run
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: {}".format(device))
config.device = device

if torch.cuda.is_available():
    print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

# set random seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# data loading
print("=> using dataset '{}'".format(config.data.dataset))
DATASET = ArtPainting_processed(config)
_, val_loader = DATASET.get_loaders(parse_patches=False)

# create model
print("=> creating denoising-diffusion model with wrapper...")
diffusion = DenoisingDiffusion(resume, image_folder, sampling_timesteps, config)
model = DiffusiveRestoration(diffusion, resume, image_folder, config)
model.restore(val_loader, validation=test_set, r=grid_r)