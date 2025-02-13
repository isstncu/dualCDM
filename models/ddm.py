import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
import sys
import logging
import csv

from models.unet_FFT import DiffusionUNet

import torch.nn.functional as F
from timeit import default_timer as timer
import pandas as pd

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

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
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
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

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def get_logger(logpath=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if logpath is not None:
            file_handler = logging.FileHandler(logpath)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
    return logger
def log_csv(filepath, values, header=None, multirows=False):
    empty = False
    if not filepath.exists():
        filepath.touch()
        empty = True

    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        if empty and header:
            writer.writerow(header)
        if multirows:
            writer.writerows(values)
        else:
            writer.writerow(values)
class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.logger = get_logger()
        self.history = args.loss_dir / 'history.csv'

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.L1 = nn.L1Loss()
        self.MSE = nn.MSELoss()

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {})".format(load_path, checkpoint['epoch']))

    def train_on_epoch(self, epoch, train_loader):
        self.model.train()
        epg_loss = AverageMeter()
        epg_error = AverageMeter()
        data_start = time.time()
        data_time = 0
        
        batches = len(train_loader)
        for i, x in enumerate(train_loader):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            n = x.size(0)
            data_time += time.time() - data_start

            x = x.to(self.device)
            
            sar_channel = 2
            e = torch.randn_like(x[:, sar_channel:, :, :])
            b = self.betas

            # antithetic sampling对偶抽样法
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            
            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x_t = x[:, sar_channel:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
            
            output = self.model(torch.cat([x[:, :sar_channel, :, :], x_t], dim=1), t.float())
            
            l1_loss = self.L1(torch.fft.fftn(output, dim=(-2, -1)),torch.fft.fftn(e, dim=(-2, -1)))
            loss = self.MSE(output, e) + 0.01*l1_loss 

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema_helper.update(self.model)
            
            epg_loss.update(loss.item())
            mse = F.mse_loss(output.detach(), e).item()
            epg_error.update(mse)
            data_start = time.time()

            print(f"Epoch[{epoch} {i}/{batches}], loss: {loss.item()}, data time: {data_time / (i+1)}")
        return epg_loss.avg, epg_error.avg
    @torch.no_grad()
    def test_on_epoch(self, epoch, val_loader):
        self.model.eval()
        epoch_error = AverageMeter()

        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.patch_size))
        print(f"Processing a single batch of validation images at epoch: {epoch}")
        for i, x in enumerate(val_loader):
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            break
        n = x.size(0)
        sar_channel = 2
        out_channel = 4
        x_cond = x[:, :sar_channel, :, :].to(self.device)
        true = x[:, sar_channel:, :, :].to(self.device)
        x = torch.randn(n, out_channel, self.config.data.patch_size, self.config.data.patch_size, device=self.device)
        x = self.sample_image(x_cond, x).to(self.device)
        
        #MSE
        val_error = F.mse_loss(x, true)
        epoch_error.update(val_error.item())

        for i in range(n):
            utils.logging.save_image(x[i], os.path.join(image_folder, str(epoch), f"{i}.tif"))
        return epoch_error.avg

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        
        least_error = float('inf')

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)
        if self.history.exists():
            df = pd.read_csv(self.history)
            least_error = df['val_error_mse'].min()
            val_error = least_error

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('Epoch:', epoch)
            t_start = timer()
            train_loss, train_error = self.train_on_epoch(epoch, train_loader)
            n_e = epoch+1
            if n_e >= 1000:
                validation_epoch = 5
            else:
                validation_epoch = 10
            if n_e % validation_epoch == 0 or epoch == 0 :
                val_error_mse = self.test_on_epoch(epoch, val_loader)
            t_end = timer()
            self.logger.info(f'Epoch[{epoch}] '
                             f'train_loss: {train_loss:.6f} - '
                             f'train_error: {train_error:.6f} - '
                             f'val_error_mse: {val_error_mse:.6f} - '
                             f'Time: {t_end - t_start}s')
            csv_header = ['epoch', 'train_loss', 'train_error', 'val_error_mse']
            csv_values = [epoch, train_loss, train_error, val_error_mse]
            log_csv(self.history, csv_values, header=csv_header)

            if val_error_mse < least_error:
                least_error = val_error_mse
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join(self.args.image_folder, 'ckpts', self.config.data.dataset + '_ddpm' + str(epoch)))
                
            elif epoch % 25 == 0:
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join(self.args.image_folder, 'ckpts', self.config.data.dataset + '_ddpm' + str(epoch)))


    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps 
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip) 
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
        if last:
            xs = xs[0][-1]
        return xs

