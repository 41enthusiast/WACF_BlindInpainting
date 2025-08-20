import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.utils as tvu
from torchvision.transforms.functional import crop
import torch.nn.functional as F
import yaml
import argparse

############################## Train and Test Pipeline ########################################

def sample_image(x_cond, x, m, model, betas, config, sampling_timesteps, last=True, patch_locs=None, patch_size=None):
        skip = config.diffusion.num_diffusion_timesteps // sampling_timesteps
        seq = range(0, config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = generalized_steps_overlapping(x, x_cond,m, seq =seq, model =model, b =betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = generalized_steps(x, x_cond, m, seq =seq, model =model, b =betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs

def sample_validation_patches(val_loader, step, image_folder, device, config, model, betas, sampling_timesteps):
    image_folder = os.path.join(image_folder, config.data.dataset + str(config.data.image_size))
    with torch.no_grad():
        print(f"Processing a single batch of validation images at step: {step}")
        for i, (x, y, m, _) in enumerate(val_loader):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            break
        n = x.size(0)
        x_cond = x[:, :3, :, :].to(device)
        x_cond = data_transform(x_cond)
        x = torch.randn(n, 3, config.data.image_size, config.data.image_size, device=device)
        x = sample_image(x_cond, x, m.to(device), model, betas, config, sampling_timesteps)
        x = inverse_data_transform(x)
        x_cond = inverse_data_transform(x_cond)

        for i in range(n):
            save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
            save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))

############################## Configuration ########################################

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    config = 'config/art_painting.yml'
    resume = 'lolol.pth'#'/content/drive/MyDrive/img_inpainting_proj/Art_Dataset_ddpm_trained.pth.tar'
    test_set = 'art_painting'
    sampling_timesteps = 25
    grid_r = 16
    seed = 61
    image_folder = 'results/images/'

    with open(config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return resume, test_set, sampling_timesteps, grid_r, seed, image_folder, new_config


############################## Logging ########################################

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)

def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path, weights_only=False)
    else:
        return torch.load(path, map_location=device, weights_only=False)
    
############################## Dataset Transformations ########################################

# For fusion of inpainted region and og background
def pad_to_multiple(image: np.ndarray, multiple: int = 8):
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

def crop_to_original(image: np.ndarray, h: int, w: int):
    return image[:h, :w]

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

############################## Transformations ########################################

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                # print(param.data.device, self.shadow[name].data.device, self.mu)
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data.to(self.device)

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            print('EMA device: ', inner_module.config.device)
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

############################## Sampling ########################################

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, x_cond, m_cond, seq, model, b, eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        ms = [m_cond]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            m_cond = m_cond.to(x.device)
            et = model(torch.cat([x_cond, xt], dim=1), t, m_cond)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds

def generalized_steps_overlapping(x, x_cond, m_cond, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        x_grid_mask = torch.zeros_like(x_cond, device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            m_cond = m_cond.to(x.device)
            et_output = torch.zeros_like(x_cond, device=x.device)
            if manual_batching:
                manual_batching_size = 64
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                m_cond_patch = torch.cat([data_transform(crop(m_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
                for i in range(0, len(corners), manual_batching_size):
                    outputs = model(torch.cat([x_cond_patch[i:i+manual_batching_size],
                                               xt_patch[i:i+manual_batching_size]], dim=1), t, m_cond_patch[i:i+manual_batching_size])
                    for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    x_cond_patch = data_transform(x_cond_patch)
                    m_cond_patch = crop(m_cond, hi, wi, p_size, p_size)
                    m_cond_patch = data_transform(m_cond_patch)
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch], dim=1), t)
            et = torch.div(et_output, x_grid_mask)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds

def noise_estimation_loss(model, x0, t, m, m_true, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1),
                   t.float(),
                   m)
    output_mg = model(torch.cat([x0[:, :3, :, :], x], dim=1),
                   t.float(),
                   m_true)
    #guidance scale factor as 0.1
    guided_out = output + 0.1*(output_mg - output)
    return (e - guided_out).square().sum(dim=(1, 2, 3)).mean(dim=0)

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    # changed dtype to float32
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float32) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float32)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float32)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float32)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)