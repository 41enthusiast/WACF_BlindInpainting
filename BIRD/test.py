from torch import nn
import torch
import os
import numpy as np
from PIL import Image
from utils import DDIMScheduler, DDIM_efficient_feed_forward, get_mask, generate_noisy_image_and_mask, process, ensure_reproducibility, load_pretrained_diffusion_model, dict2namespace
import yaml

### Reproducibility
torch.set_printoptions(sci_mode=False)
ensure_reproducibility(0)

with open( "config/celeba_hq.yml", "r") as f:
    config1 = yaml.safe_load(f)
config = dict2namespace(config1)
model, device = load_pretrained_diffusion_model(config)

### Define the DDIM scheduler
mask_path = '../../data/art_painting/test/mask'
img_path = '../../data/art_painting/test/image'
for test_img in os.listdir(img_path):
  print(test_img)
  ddim_scheduler=DDIMScheduler(beta_start=config.diffusion.beta_start, beta_end=config.diffusion.beta_end, beta_schedule=config.diffusion.beta_schedule)
  ddim_scheduler.set_timesteps(config.diffusion.num_diffusion_timesteps // 100)


  mask = get_mask(f'{mask_path}/{test_img}')
  _, img_np, __ = generate_noisy_image_and_mask(f'{img_path}/{test_img}')
  img_torch = torch.tensor(img_np).permute(2,0,1).unsqueeze(0)
  print(mask.shape)
  t_mask = mask.unsqueeze(0).cuda()
  radii =  torch.ones([1, 1, 1]).cuda() * (np.sqrt(config.data.image_size*config.data.image_size*config.model.in_channels))

  latent = torch.nn.parameter.Parameter(torch.randn( 1, config.model.in_channels, config.data.image_size, config.data.image_size).to(device))
  l2_loss=nn.MSELoss() #nn.L1Loss()
  optimizer = torch.optim.Adam([{'params':latent,'lr':0.01}])#


  for iteration in range(200):
      optimizer.zero_grad()
      x_0_hat = DDIM_efficient_feed_forward(latent, model, ddim_scheduler)
      loss = l2_loss(x_0_hat*t_mask, img_torch.cuda()*t_mask)
      loss.backward()
      optimizer.step()

      #Project to the Sphere of radius sqrt(D)
      for param in latent:
          param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
          param.data.mul_(radii)

  #psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
  #print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
  Image.fromarray(process(x_0_hat, 0)).save('results/'+test_img)

