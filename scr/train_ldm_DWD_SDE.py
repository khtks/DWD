import sde
import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import einops
from torch.utils._pytree import tree_map
# import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import libs.autoencoder

import shutil
import yaml
import argparse
import tools.utils as utils

from tools.utils import *

from dotted_dict import DottedDict
from pathlib import Path
import time
from libs.ldm.util import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR
from libs.ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from libs.iddpm.unet import EncoderUNetModel, UNetModel
from libs.ldm.modules.ema import LitEma
from libs.PU_loss import *
import copy
from datetime import date

from cyanure.data_processing import preprocess
from cyanure.estimators import Classifier
from libs.resnet_discriminator import ResNetImageDiscriminator
import torchvision

parser = argparse.ArgumentParser(description='Arguments of training latent diffusion for Semi-Diffusion')
parser.add_argument('--numworkers', type=int, default=4, help='num workers for training Unet model')
parser.add_argument('--T', type=int, default=1000, help='timesteps for Unet model')
parser.add_argument('--droprate', type=float, default=0.1, help='dropout rate for model')
parser.add_argument('--dtype', default=torch.float32)
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')

# Path config
parser.add_argument("--prefix", default='assets/DWD', type=str)
parser.add_argument('--moddir', type=str, default='./assets/DWD/ckpt', help='model addresses')
parser.add_argument('--samdir', type=str, default='./assets/DWD/samples', help='sample addresses')
parser.add_argument('--config_path', type=str, default='LDM_KL_Cifars', help='config path for train')
parser.add_argument('--ae_config_path', type=str, default='AE_KL_Cifars_16x16x3_Attn', help='config path for train')
parser.add_argument('--ckpt_path', type=str, default=None, help='config path for train')

# Train loop config
parser.add_argument('--epochs', type=int, default=1500, help='total epochs for training')
parser.add_argument('--start_epoch', type=int, default=0, help='start epochs for training')
parser.add_argument('--interval', type=int, default=50, help='epochs interval for evaluation')
parser.add_argument('--steps', type=int, default=256, help='train steps per epoch')

# Generation config
parser.add_argument('--clsnum', type=int, default=10, help='num of label classes')
parser.add_argument('--genbatch', type=int, default=10, help='batch size for sampling process')

# Training config
parser.add_argument('--batch_size', type=int, default=16, help='batch size per device for training Unet model')
parser.add_argument('--mu', default=7, type=int)
parser.add_argument('--ae_method', default='kl', type=str, choices=['kl', 'vq'])
parser.add_argument('--sample', default='multiple', type=str, choices=['multiple', 'random'])
parser.add_argument('--cond', required=True, default='joint', type=str, choices=['multi', 'joint', 'pseudo', 'uncond'])
parser.add_argument('--train_cond', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--pseudo', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--ema', default=True, type=lambda s: s in ['True', 'true', 1])

parser.add_argument('--rep_first', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--rep_with_img', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--force_save', default=False, type=lambda s: s in ['True', 'true', 1])

parser.add_argument('--cfg', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--p_uncond', default=0.1, type=float)
parser.add_argument('--w', type=float, default=0.4, help='hyperparameters for classifier-free guidance strength')
parser.add_argument('--threshold', type=float, default=0.1, help='threshold for classifier-free guidance')
parser.add_argument('--flip', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--u_data', default='ood', type=str, choices=['ood', 'gen'])
parser.add_argument('--dataset', default='cifars', type=str, choices=['cifars', 'six'])

# Discriminator config
parser.add_argument('--disc_arch', default='resnet', type=str, choices=['resnet'])
parser.add_argument('--prior', default=0.3, type=float)
parser.add_argument('--pu_loss', default='nnPU', type=str, choices=['nnPU', 'nnPUSB'])
parser.add_argument('--IS', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--alpha', default=3, type=int)

#  General config
parser.add_argument('--exp_name', type=str, default=None, required=True)
parser.add_argument('--resume', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--FT', default=False, type=lambda s: s in ['True', 'true', 1])



@torch.no_grad()
def get_rep_first(AE, labeled_dataset, unlabeled_dataset, config):
    print("Target: utils.get_rep_first")
    if os.path.exists(config.data.rep_path) and not config.force_save:
        print(f"Load latents from {config.data.rep_path}")
        extracted = torch.load(config.data.rep_path)
        l_reps, l_labs, u_reps, u_labs = extracted['labeled']['rep'], extracted['labeled']['label'], \
                                         extracted['unlabeled']['rep'], extracted['unlabeled']['label']

        # del AE.scale_factor
        # AE.register_buffer('scale_factor', extracted['scale_factor'])
        # print(f"At the very first step, setting AE.scale_factor to {AE.scale_factor} for pre-extracted latent")

    else:
        labeled_loader = DataLoader(labeled_dataset, batch_size=config.lbs, shuffle=True, drop_last=False,
                                    num_workers=4, pin_memory=True, persistent_workers=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.ubs, shuffle=True, drop_last=False,
                                      num_workers=4, pin_memory=True, persistent_workers=True)

        l_reps, l_labs = torch.FloatTensor(), torch.LongTensor()
        u_reps, u_labs = torch.FloatTensor(), torch.LongTensor()

        (lx, _), (ux, _) = next(iter(labeled_loader)), next(iter(unlabeled_loader))
        x = torch.cat((lx, ux), dim=0).cuda()
        z = AE.encode(x).sample()

        print("\n# Extract Labeled data")
        for img, lab in tqdm(labeled_loader):
            z = AE.encode(img.cuda()).sample()

            l_reps = torch.cat((l_reps, z.detach().cpu()), dim=0)
            l_labs = torch.cat((l_labs, lab.detach().cpu()), dim=0)

        print("# Extract Unlabeled data")
        for img, lab in tqdm(unlabeled_loader):
            z = AE.encode(img.cuda()).sample()

            u_reps = torch.cat((u_reps, z.detach().cpu()), dim=0)
            u_labs = torch.cat((u_labs, lab.detach().cpu()), dim=0)

        torch.save({"labeled": {"rep": l_reps, "label": l_labs}, "unlabeled": {"rep": u_reps, "label": u_labs},
                    "scale_factor": AE.scale_factor}, config.data.rep_path)
        print(f"Saving extracted latents on {config.data.rep_path}")

    labeled_dataset = RepDataset(reps=l_reps, labs=l_labs,
                                 dataset=list(np.random.permutation(np.arange(l_reps.shape[0]))))
    unlabeled_dataset = RepDataset(reps=u_reps, labs=u_labs,
                                   dataset=list(np.random.permutation(np.arange(u_reps.shape[0]))))

    return labeled_dataset, unlabeled_dataset


def set_config(args):
    with open(f'./configs/{args.config_path}.yaml', 'r') as y_file:
        yaml_file = yaml.load(y_file, Loader=yaml.FullLoader)
        config = DottedDict(dict(yaml_file))

    with open(f'./configs/{args.ae_config_path}.yaml', 'r') as y_file:
        yaml_file = yaml.load(y_file, Loader=yaml.FullLoader)
        ae_config = DottedDict(dict(yaml_file))
        config.model.params.first_stage_config.params.ddconfig = ae_config.model.params.ddconfig
        config.model.params.first_stage_config.params.embed_dim = ae_config.model.params.embed_dim
        config.data.z_shape = ae_config.data.embed_dim

    keys = list(args.__dict__.keys())
    values = list(args.__dict__.values())
    [setattr(config, keys[i], values[i]) for i in range(len(keys))]

    config.model.params.cond_stage_trainable = args.train_cond
    config.context_dim = config.model.params.cond_stage_config.params.embed_dim

    config.exp_name = f"DWD_{config.exp_name}_{config.cond.capitalize()}"
    config.path = DottedDict(dict(workdir=Path(config.prefix, config.exp_name), moddir=Path(config.prefix, config.exp_name, 'ckpt'),
                                  samdir=Path(config.prefix, config.exp_name, 'samples')))

    for i, (k, v) in enumerate(config.path.items()):
        if i > 0 and (v.exists() and not config.resume and len(os.listdir(v)) > 1):
            v = Path(v, f'{date.today().strftime(f"%y.%m.%d")}')
            setattr(config.path, k, v)
        print("{:>7s} : {}".format(k, v))
        v.mkdir(parents=True, exist_ok=True)

    if config.rep_first:
        Path(config.path.workdir, 'assets', 'reps').mkdir(parents=True, exist_ok=True)
        config.data.rep_path = Path(config.path.workdir, 'assets', 'reps', f"{config.data.name}_{config.u_data}_{config.exp_name}.pt")

    if config.general.benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.device = 'cuda'

    return config, ae_config


def resuming(path, models, optimizers, schedulers, config):
    if config.resume or config.FT:
        assert os.path.exists(path)
        print("==> Load file from:", path)
        checkpoint = torch.load(path)

        global global_step
        global_step = checkpoint['global_step']
        config.start_epoch = checkpoint['epoch']

        optimizer, d_optimizer = optimizers
        scheduler, d_scheduler = schedulers
        nnet, nnet_ema, AE, discriminator = models

        m1 = nnet.load_state_dict(checkpoint['nnet'])
        m2 = nnet_ema.load_state_dict(checkpoint['nnet_ema'])
        me = AE.load_state_dict(checkpoint['AE'], strict=False)

        if config.FT:
            assert me.unexpected_keys == ['scale_factor'], me.unexpected_keys

        optimizer.load_state_dict(checkpoint['optimizer'])

        if config.resume:
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])
            d_scheduler.load_state_dict(checkpoint['d_scheduler'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            discriminator.load_state_dict(checkpoint['disc'])
        else:
            _, scheduler, _, _ = init_train_setting(nnet, discriminator, config)

    return None


# @contextmanager
def ema_scope(nnet, nnet_ema):
    nnet_ema.store(nnet.parameters())
    nnet_ema.copy_to(nnet)
    nnet_ema.restore(nnet.parameters())


def on_train_batch_end(nnet_ema, nnet):
    nnet_ema(nnet)


def init_train_setting(nnet, discriminator, config):
    opt_config = config.optimizer

    lr = opt_config.lr
    params = list(nnet.parameters())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.003)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=1e-4)

    scheduler_config = config.model.params.scheduler_config
    scheduler_config.params.cycle_lengths = [config.steps * config.epochs]
    scheduler = instantiate_from_config(scheduler_config)

    print("Setting up LambdaLR scheduler...")
    d_scheduler = LambdaLR(d_optimizer, lr_lambda=scheduler.schedule)
    scheduler = LambdaLR(optimizer, lr_lambda=scheduler.schedule)

    return optimizer, scheduler, d_optimizer, d_scheduler


def init_dataset(AE, config):
    config.data.params = dict(cfg=config.cfg, random_flip=config.flip, p_uncond=config.p_uncond, u_data=config.u_data)

    dataset = get_dataset(name=config.data.name, **config.data.params)
    config.fid_stat = Path(dataset.fid_stat)
    assert config.fid_stat.exists()

    labeled_dataset = dataset.get_split(split='train', labeled=cond)
    unlabeled_dataset = dataset.get_split(split='unlabeled', labeled=cond)

    if config.rep_first:
        labeled_dataset, unlabeled_dataset = get_rep_first(AE, labeled_dataset, unlabeled_dataset, config)
        if config.cfg:
            labeled_dataset = CFGDataset(labeled_dataset, config.p_uncond, dataset.K)
            unlabeled_dataset = CFGDataset(unlabeled_dataset, config.p_uncond, dataset.K)

    labeled_loader = DataLoader(labeled_dataset, batch_size=config.lbs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.ubs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)

    return labeled_loader, unlabeled_loader


def init_score_model(nnet, nnet_ema, config):
    if cond and config.cfg and config.w > 0:
        cls_num = config.clsnum

        def cfg_nnet(x, timesteps, c, get_rep=False):
            with nnet_ema.ema_scope(nnet):
                # Conditional
                kwargs = {'y': c}
                _cond = nnet(x, timesteps, **kwargs)

                # Unconditional
                kwargs = {'y': torch.tensor([cls_num] * c.size(0)).cuda()}
                _uncond = nnet(x, timesteps, **kwargs)
            return _cond + config.w * (_cond - _uncond)

        score_model = sde.ScoreModel(nnet, pred=config.general.pred, sde=sde.VPSDE())
        score_model_ema = sde.ScoreModel(cfg_nnet, pred=config.general.pred, sde=sde.VPSDE())

    else:
        score_model = sde.ScoreModel(nnet, pred=config.general.pred, sde=sde.VPSDE())
        with nnet_ema.ema_scope(nnet):
            score_model_ema = sde.ScoreModel(nnet, pred=config.general.pred, sde=sde.VPSDE())

    return score_model, score_model_ema


def init_model(config):
    from libs.ldm.util import instantiate_from_config

    config.model.params.first_stage_config.target = 'libs.autoencoder.AutoencoderKL'
    AE = instantiate_from_config(config.model.params.first_stage_config).cuda()
    for param in AE.parameters():
        param.requires_grad = False

    if cond:
        config.model.params.unet_config.params.num_classes = config.clsnum + 1
    else:
        config.model.params.unet_config.params.num_classes = None

    unet_config = config.model.params.unet_config
    ae_config = config.model.params.first_stage_config

    unet_config.params.in_channels = ae_config.params.embed_dim
    unet_config.params.out_channels = ae_config.params.embed_dim
    unet_config.params.context_dim = None
    unet_config.params.use_spatial_transformer = False

    nnet = instantiate_from_config(config.model.params.unet_config).cuda()
    nnet_ema = LitEma(nnet)
    print(f"Keeping EMAs of {len(list(nnet_ema.buffers()))}.")

    proj_in, in_ch, proj_dim = 384, config.data.z_shape[0], 128
    disc_model = ResNetImageDiscriminator
    discriminator = disc_model(in_ch=in_ch, proj_in=proj_in, proj_dim=proj_dim, ema=config.ema, config=config).cuda()

    AE.eval()
    nnet_ema.eval()
    return nnet, nnet_ema, AE, discriminator


def get_data_generator(labeled_loader, unlabeled_loader, config):
    device = config.device
    while True:
        imgs = torch.FloatTensor().cuda()
        labs = torch.LongTensor().cuda()

        try:
            img_l, lab_l = next(labeled_iter)
        except:
            labeled_iter = iter(labeled_loader)
            img_l, lab_l = next(labeled_iter)
        finally:
            img_l, lab_l = img_l.cuda(), lab_l.long().cuda()
            imgs, labs = torch.cat((imgs, img_l), dim=0).to(device), torch.cat((labs, lab_l), dim=0).to(device)

        try:
            img_u, lab_u = next(unlabeled_iter)
        except:
            unlabeled_iter = iter(unlabeled_loader)
            img_u, lab_u = next(unlabeled_iter)
        finally:
            img_u, lab_u = img_u.cuda(), lab_u.cuda()
            imgs, labs = torch.cat((imgs, img_u), dim=0).to(device), torch.cat((labs, lab_u), dim=0).to(device)

        batch = dict(image=imgs, class_label=labs)
        yield batch


def get_input(AE, data_generator, batch_idx, config):
    batch = next(data_generator)

    if config.rep_with_img:
        x, c = batch['image'], batch['class_label']
        z = AE.encode(x.cuda()).sample()

        if (config.epoch == config.start_epoch) and batch_idx == 0:
            del AE.scale_factor
            AE.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"At the very first step, setting LDM.scale_factor to {AE.scale_factor}")

        z = AE.scale_factor * z
        return x, z, c

    if config.rep_first:
        z, c = batch['image'], batch['class_label']

        if (config.epoch == config.start_epoch) and batch_idx == 0:
            del AE.scale_factor
            AE.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"At the very first step, setting AE.scale_factor to {AE.scale_factor} for pre-extracted latent")

    else:
        x, c = batch['image'], batch['class_label']
        z = AE.encode(x.cuda()).sample()

        if (config.epoch == config.start_epoch) and batch_idx == 0:
            del AE.scale_factor
            AE.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"At the very first step, setting LDM.scale_factor to {AE.scale_factor}")

    if not cond:
        c = torch.tensor([config.clsnum] * c.size(0)).cuda()

    return z, c


def validation_step(score_model_ema, models, z_shape, config, algorithm='dpm_solver', save_files=False):
    nnet, nnet_ema, AE, _ = models
    nnet.eval()
    nnet_ema.eval()
    AE.eval()

    def sample_fn(n_samples, labs=None):
        _z_init = torch.randn(n_samples, *z_shape, device=config.device).detach()

        if cond:  # Conditional
            kwargs = {'c': labs}
        else:  # Unconditional
            kwargs = {}

        noise_schedule = NoiseScheduleVP(schedule='linear')
        model_fn = model_wrapper(score_model_ema.noise_pred, noise_schedule, time_input_type='0', model_kwargs=kwargs)
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        _z = dpm_solver.sample(_z_init, steps=10, eps=1e-3, order=1, adaptive_step_size=False, fast_version=True, )

        z = 1. / AE.scale_factor * _z
        return AE.decode(_z)

    target_dir = Path(config.path.samdir, f'fid_target')
    target_dir.mkdir(parents=True, exist_ok=True)

    img_size = config.data.img_size
    generated = torch.FloatTensor().cuda()
    labs = torch.ones(config.genbatch, dtype=torch.long) * torch.arange(start=0, end=config.clsnum).reshape(-1, 1)
    labs = labs.to(config.device)

    for index in tqdm(range(config.clsnum), desc='sample2dir'):
        lab = labs[index]
        samples = unpreprocess(sample_fn(config.genbatch, labs=lab))
        generated = torch.cat((generated, samples), dim=0)

        for idx, sample in enumerate(samples):
            save_image(sample, os.path.join(target_dir, f"{index}_{idx}.png"))

    lpips = get_lpips(generated, generated, config.n_samples, img_size)
    fid = calculate_fid_given_paths((config.fid_stat, target_dir))

    if save_files:
        img = generated.reshape(config.clsnum, config.genbatch, 3, img_size, img_size).contiguous()
        img = img.reshape(config.n_samples, 3, img_size, img_size)
        save_image(img, os.path.join(config.path.samdir,
                                     f'generated_{config.epoch + 1}epc_{config.exp_name}_{int(np.round(fid))}_FID.png'),
                   nrow=config.genbatch)

    return fid.item(), lpips


def train_step(score_model, models, data_generator, optimizers, schedulers, config):
    global global_step, ths

    nnet, nnet_ema, AE, discriminator = models
    nnet.train()
    discriminator.train()

    optimizer, d_optimizer = optimizers
    scheduler, d_scheduler = schedulers

    print("\nEpoch: [{}]  Start Training Step".format(config.epoch + 1))
    losses = AverageMeter('')
    mse_losses = AverageMeter('')
    disc_losses = AverageMeter('')

    pbar = tqdm(total=args.steps)
    pu_fn = nnPUloss if config.pu_loss == 'nnPU' else nnPUSBloss
    pu_loss = pu_fn(config=config, IS=config.IS)

    for batch_idx in range(args.steps):
        inputs = get_input(AE, data_generator, batch_idx, config)

        if config.rep_with_img: x, z, c = inputs
        else: z, c = inputs
        t = torch.rand((z.shape[0]), device=z.device)

        #########################
        #  Train Discriminator  #
        #########################

        indices = torch.randperm(config.bs)
        pu_target = torch.tensor([(1. if i < config.lbs else 0.) for i in range(config.bs)]).cuda()
        z, c, t, pu_target = z[indices, :], c[indices], t[indices], pu_target[indices]
        if config.rep_with_img: x = x[indices, :]

        out = discriminator(z)
        d_loss, coeff, out_l, out_u, pu_indices = pu_loss(out, pu_target)
        u_coeff = coeff[pu_indices[1]].detach()

        disc_losses.update(d_loss.item())
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        ####################
        # Train Diffusion  #
        ####################

        if cond: kwargs = {'y': c}
        else: kwargs = {}
        loss, mse_loss = sde.LSimple_with_coeff(score_model=score_model, x0=z, t=t, pred='noise_pred', u_coeff=u_coeff, lbs=config.lbs, **kwargs)

        losses.update(loss.item())
        mse_losses.update(mse_loss.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(global_step)
        d_scheduler.step(global_step)

        txt = (f"Epoch: [{config.epoch + 1:>4d}/{args.epochs:>4d}] "
               f"Train Iter: {global_step + 1:3}/{args.steps * args.epochs:4}. "
               f"loss: {losses.avg:>.5f}. mse_loss: {mse_losses.avg:>4f}. disc_loss: {disc_losses.avg:>.4f}. "
               f"l_out: {pu_loss.l_outs.avg:>.4f}. u_out: {pu_loss.u_outs.avg:>.4f}. ")

        pbar.set_description(txt)
        pbar.update()

        global_step += 1
        on_train_batch_end(nnet_ema, nnet)

    loss_dict = [losses.avg, mse_losses.avg, disc_losses.avg, pu_loss.l_outs.avg, pu_loss.u_outs.avg]

    return loss_dict


def main(config):
    config.lbs = config.batch_size
    config.ubs = config.batch_size * config.mu
    config.bs = config.lbs + config.ubs

    global global_step
    global_step = 0

    # Initialize Model
    nnet, nnet_ema, AE, discriminator = init_model(config)
    models = [nnet, nnet_ema, AE, discriminator]

    # Initialize train setting (Optimizer, Scheduler)
    optimizer, scheduler, d_optimizer, d_scheduler = init_train_setting(nnet, discriminator, config)
    optimizers, schedulers = [optimizer, d_optimizer], [scheduler, d_scheduler]

    # Resuming or Fine-tuning

    model_name = 'Cifar100_16x16x3_Attn_Concat_Uncond/ckpt_1000_Cifars_16x16x3_Attn_Concat_Uncond_190_FID.pt'
    path = os.path.join(config.moddir, model_name)
    resuming(path, models, optimizers, schedulers, config)

    # Initialize Dataset
    labeled_loader, unlabeled_loader = init_dataset(AE, config)
    data_generator = get_data_generator(labeled_loader, unlabeled_loader, config)

    # set the score_model to train
    score_model, score_model_ema = init_score_model(nnet, nnet_ema, config)

    best_fid = np.inf
    best_lpips = -np.inf

    for config.epoch in range(config.start_epoch, config.epochs):
        is_best = False
        save_files = ((config.epoch + 1) % config.interval == 0)

        # Train Step
        outputs = train_step(score_model, models, data_generator, optimizers, schedulers, config)
        loss, mse_loss, disc_loss, l_out, u_out = outputs

        if save_files:
            # Validation Step
            val_fid, val_lpips = validation_step(score_model_ema, models, config.data.z_shape, config, algorithm='dpm_solver', save_files=save_files)

            # Logging
            print(f'Epoch: [{config.epoch + 1:3}/{args.epochs:3}]  loss: {mse_loss + disc_loss:.5f}  mse_loss: {mse_loss:.5f}  disc_loss: {disc_loss:.5f}  fid: {val_fid:.5f}  lpips: {val_lpips:.5f}')

            if val_fid < best_fid:
                is_best = True
                best_fid = val_fid

            if val_lpips > best_lpips:
                best_lpips = val_lpips

            if is_best or save_files:
                checkpoint = {
                    'nnet': nnet.state_dict(),
                    'nnet_ema': nnet_ema.state_dict(),
                    'AE': AE.state_dict(),
                    'disc': discriminator.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'd_scheduler': d_scheduler.state_dict(),
                    'epoch': config.epoch + 1,
                    'global_step': global_step,
                }

                if is_best:
                    torch.save(checkpoint, os.path.join(config.path.moddir, f'{config.exp_name}_best.pt'))
                    print(f"Epoch: [{config.epoch + 1}]  Saving best model..")

                if save_files:
                    # save_checkpoint(checkpoint, False, config.path.moddir, f'{config.exp_name}_last.pt')
                    save_checkpoint(checkpoint, False, config.path.moddir,
                                    f'ckpt_{config.epoch + 1:04d}_{config.exp_name}_{int(np.round(val_fid))}_FID.pt')

    print(f'Best fid: {best_fid:.5f}  lpips: {best_lpips:.5f}\n')
    print("...DONE")


if __name__ == "__main__":
    global global_step

    args = parser.parse_args()
    config, ae_config = set_config(args)
    gap = torch.nn.AdaptiveAvgPool2d((1, 1))

    config.n_samples = config.clsnum * config.genbatch
    cond = (config.cond != 'uncond')

    main(config)
