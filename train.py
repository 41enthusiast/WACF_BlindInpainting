from utils import data_transform, sample_validation_patches, parse_args_and_config, get_beta_schedule, get_optimizer, save_checkpoint, noise_estimation_loss, EMAHelper
from model import DiffusionUNet
from dataset import ArtPainting_processed
import time
import os
import torch

data_start = time.time()
data_time = 0
cross_attn_scores = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resume, test_set, sampling_timesteps, grid_r, seed, image_folder, config = parse_args_and_config()
model = DiffusionUNet(config).to(device)
# model, cross_attn_scores = unet_store_cross_attention_scores(model, cross_attn_scores)
step, start_epoch = 0, 0
betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
betas = torch.from_numpy(betas).float().to(device)
num_timesteps = betas.shape[0]
optimizer = get_optimizer(config, model.parameters())
ema_helper = EMAHelper()
ema_helper.register(model)

DATASET = ArtPainting_processed(config)
tdl, vdl = DATASET.get_loaders()

for epoch in range(start_epoch, config.training.n_epochs):
    print('epoch: ', epoch)
    data_start = time.time()
    data_time = 0

    for i, (x, y, m, m_true) in enumerate(tdl):
        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
        n = x.size(0)
        data_time += time.time() - data_start
        model.train()
        step += 1

        x = x.to(device)
        x = data_transform(x)
        m = m.to(device) if m is not None else None
        m_true = m_true.to(device) if m_true is not None else None
        e = torch.randn_like(x[:, 3:, :, :])
        b = betas

        # antithetic sampling
        t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
        t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
        loss = noise_estimation_loss(model, x, t, m, m_true, e, b)

        #attention guidance with real masks
        attn_map = model.down[2].attn[0].get_attention_scores
        m_true = m_true.view(-1, 1, m_true.shape[-2], m_true.shape[-1])
        m_true = torch.nn.functional.interpolate(m_true, size=attn_map.shape[-2:], mode='nearest')
        m_true = m_true.expand(-1, attn_map.shape[1], -1, -1)


        if step % 10 == 0:
            print(f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}")
            # break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_helper.update(model)
        data_start = time.time()

        if step % config.training.validation_freq == 0:
            print('Validatng...')
            model.eval()
            sample_validation_patches(vdl, step)

        if step % config.training.snapshot_freq == 0 or step == 1:
            print('Saving checkpoints')
            save_checkpoint({
                'epoch': epoch + 1,
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'ema_helper': ema_helper.state_dict(),
                'config': config
            }, filename=os.path.join(config.data.data_dir, 'ckpts', config.data.dataset + '_ddpm'))