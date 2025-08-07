from torch.utils.data import DataLoader
import yaml
from utils import dict2namespace, DenoisingDiffusion
import torch
from dataset.artpainting_processed import ArtPainting_processed
import numpy as np

def parse_args_and_config():
    config = 'allweather.yml'
    resume = 'Art_Dataset_ddpm.pth.tar'
    test_set = 'art_painting'
    sampling_timesteps = 25
    grid_r = 16
    seed = 61
    image_folder = 'results/images/'

    with open(config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return resume, test_set, sampling_timesteps, grid_r, seed, image_folder, new_config


resume, test_set, sampling_timesteps, grid_r, seed, image_folder, config = parse_args_and_config()

# setup device to run
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: {}".format(device))
config.device = device

# set random seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# data loading
print("=> using dataset '{}'".format(config.data.dataset))
DATASET = ArtPainting_processed(config)

# create model
print("=> creating denoising-diffusion model...", device)
# device
diffusion = DenoisingDiffusion(resume, image_folder, sampling_timesteps, config)
diffusion.train(DATASET)