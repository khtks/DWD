import sde
import ml_collections
import torch
from torch.optim import AdamW, Adam, SGD
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import tools.utils as utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
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
from tools.utils import *

from dotted_dict import DottedDict
from pathlib import Path
import time
from libs.ldm.util import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR
from libs.ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
import wandb
import copy
from libs.ddpm import LatentDiffusion, get_LDM
import pandas as pd
import PIL.Image as im
from libs.ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from libs.ldm.modules.ema import LitEma
from libs.modules import ClassEmbedder

# several hyperparameters for model
parser = argparse.ArgumentParser(description='Arguments of training latent diffusion for Semi-Diffusion')
parser.add_argument('--batch_size', type=int, default=64, help='batch size per device for training Unet model')
parser.add_argument('--numworkers', type=int, default=4, help='num workers for training Unet model')
parser.add_argument('--T', type=int, default=1000, help='timesteps for Unet model')
parser.add_argument('--droprate', type=float, default=0.1, help='dropout rate for model')
parser.add_argument('--dtype', default=torch.float32)
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--threshold', type=float, default=0.1, help='threshold for classifier-free guidance')

# Path config
parser.add_argument('--moddir', type=str, default='', help='model addresses')
parser.add_argument('--samdir', type=str, default='', help='sample addresses')
parser.add_argument("--prefix", default='assets/DWD', type=str)
parser.add_argument('--config_path', type=str, default='LDM_KL_Cifars', help='config path for train')
parser.add_argument('--ae_config_path', type=str, default='AE_KL_Cifars_16x16x3_Attn', help='config path for train')

# Train loop config
parser.add_argument('--epochs', type=int, default=1500, help='total epochs for training')
parser.add_argument('--start_epoch', type=int, default=0, help='start epochs for training')
parser.add_argument('--interval', type=int, default=50, help='epochs interval for evaluation')
parser.add_argument('--steps', type=int, default=256, help='train steps per epoch')

# Generation config
parser.add_argument('--clsnum', type=int, default=10, help='num of label classes')
parser.add_argument('--genbatch', type=int, default=10, help='batch size for sampling process')

# Training config
parser.add_argument('--ae_method', default='kl', type=str, choices=['kl', 'vq'])
parser.add_argument('--mu', default=4, type=int)
parser.add_argument('--train_cond', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--ema', default=True, type=lambda s: s in ['True', 'true', 1])

parser.add_argument('--rep_first', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--force_save', default=False, type=lambda s: s in ['True', 'true', 1])

parser.add_argument('--cfg', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--w', type=float, default=0.5)
parser.add_argument('--p_uncond', default=0.1, type=float)
parser.add_argument('--flip', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--u_data', default='cifars', type=str, choices=['none', 'id', 'ood', 'gen', 'six', 'cifars'])

parser.add_argument('--ipc', type=int, default=100, help='number of labeled data samples per class')
parser.add_argument('--cond', default='multi', type=str, choices=['multi', 'joint', 'pseudo', 'uncond'])
parser.add_argument('--pseudo', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--diff', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)
parser.add_argument('--dist', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)

args = parser.parse_args()

Cifar10_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

total_steps = 1000
tseq = list(np.linspace(0, 1000 - 1, total_steps).astype(int))
gap = torch.nn.AdaptiveAvgPool2d((1, 1))

data_labels = []


class CifarDataset(Dataset):
    def __init__(self, df):
        self.df = df

        normal_mean = (0.5, 0.5, 0.5)
        normal_std = (0.5, 0.5, 0.5)
        img_size = 32
        self.transpose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normal_mean, std=normal_std)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        contents = self.df.loc[idx]
        path = contents['path']

        img = im.open(path).convert('RGB')
        label = contents['label']

        x = self.transpose(img)
        y = torch.tensor(label)

        return x, y


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_config(args):
    with open(f'./configs/{args.config_path}.yaml', 'r') as y_file:
        yaml_file = yaml.load(y_file, Loader=yaml.FullLoader)
        config = DottedDict(dict(yaml_file))

    with open(f'./configs/{args.ae_config_path}.yaml', 'r') as y_file:
        yaml_file = yaml.load(y_file, Loader=yaml.FullLoader)
        ae_config = DottedDict(dict(yaml_file))
        config.model.params.first_stage_config.params.ddconfig = ae_config.model.params.ddconfig
        config.model.params.first_stage_config.params.embed_dim = ae_config.model.params.embed_dim
        config.ckpt_path = config.model.params.first_stage_config.params.ckpt_path[:-3]

    keys = list(args.__dict__.keys())
    values = list(args.__dict__.values())
    [setattr(config, keys[i], values[i]) for i in range(len(keys))]
    config.model.params.cond_stage_trainable = args.train_cond
    config.device = torch.device("cuda")

    return config, ae_config


def assign_using_dist(l_rep, u_rep, norm=True, metric='min', config=None):
    l_rep = torch.flatten(gap(l_rep), 1)
    u_rep = torch.flatten(gap(u_rep), 1)

    if norm:
        l_rep = torch.nn.functional.normalize(l_rep)
        u_rep = torch.nn.functional.normalize(u_rep)

    if metric == 'min':
        pdist = torch.cdist(u_rep, l_rep)
        min_dist, min_indices = torch.min(pdist, dim=1)
        labs = data_labels[min_indices]
    else:
        pdist = torch.FloatTensor().cuda()
        for idx in range(10):
            dist = torch.mean(torch.cdist(u_rep, l_rep[idx * 100: (idx + 1) * 100]), dim=1)
            pdist = torch.cat((pdist, dist.reshape(-1, 1)), dim=1)

        min_dist, min_indices = torch.min(pdist, dim=1)
        labs = min_indices
    return labs


def snn(query, supports, labels=None, norm=True, return_sharp=False):
    """ Soft Nearest Neighbours similarity classifier """
    # Step 1: normalize embeddings
    tau, T = 0.1, 0.25
    softmax = torch.nn.Softmax(dim=1)
    gap = torch.nn.AdaptiveAvgPool2d((1, 1))

    query = torch.flatten(gap(query), 1)
    supports = torch.flatten(gap(supports), 1)

    if norm:
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

    if labels is None: labels = np.arange(0, supports.shape[0]) // 100

    labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=10).float().cuda()
    supports, query, labels = supports.detach(), query.detach(), labels.detach()

    # Step 2: compute similarity between local embeddings
    probs = softmax(query @ supports.T / tau)
    soft_p = probs @ labels

    # Step 3: Sharpening the probability
    sharp_p = soft_p ** (1. / T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)

    if return_sharp: return sharp_p
    else: return torch.argmax(sharp_p, dim=1)


def get_pseudo_label(AE, inputs_z, config, supports, labels=None, norm=True):
    if config.cond:
        if config.pseudo:
            if config.dist:
                labs = assign_using_dist(l_rep=supports, u_rep=inputs_z, norm=norm, config=config)
            else:
                labs = snn(query=inputs_z, supports=supports, return_sharp=False)

            labs = labs.long().cuda()

        else:  # The case of Random conditioning
            labs = torch.ones(config.clsnum, len(inputs_z) // config.clsnum).type(torch.long) * torch.arange(start=0, end=config.clsnum).reshape(-1, 1)
            labs = labs.reshape(-1, 1).squeeze().to(config.device)
            labs = torch.randint(0, 10, labs.shape).to(config.device)

    else:  # The case of Unconditioned generation
        labs = torch.ones(config.clsnum, len(inputs_z) // config.clsnum).type(torch.long) * torch.arange(start=0, end=config.clsnum).reshape(-1, 1)
        labs = labs.reshape(-1, 1).squeeze().to(config.device)

    return inputs_z, labs


def get_labeled_latent(AE, loader, config):
    global data_labels

    reps = torch.FloatTensor().cuda()
    for (x, y) in loader:
        x, y = x.cuda(), y.cuda()

        rep = AE.encode(x).sample()
        reps = torch.cat((reps, rep.detach()), dim=0)
        data_labels.extend(y.tolist())

    data_labels = torch.tensor(data_labels).cuda()

    return reps


def load_checkpoint(nnet, nnet_ema, AE, model_path, loaders):
    checkpoint = torch.load(model_path, map_location=config.device)
    nnet.load_state_dict(checkpoint['nnet'])
    nnet_ema.load_state_dict(checkpoint['nnet_ema'])

    AE.load_state_dict(checkpoint['AE'], strict=False)
    if 'scale_factor' in checkpoint['AE'].keys():
        del AE.scale_factor
        AE.register_buffer('scale_factor', checkpoint['AE']['scale_factor'])

    else:
        label_loader, data_loader = loaders
        (lx, _), (ux, _) = next(iter(label_loader)), next(iter(data_loader))
        x = torch.cat((lx, ux), dim=0).cuda()

        z = AE.encode(x).sample()

        del AE.scale_factor
        AE.register_buffer('scale_factor', 1. / z.flatten().std())
        print(f"At the very first step, setting AE.scale_factor to {AE.scale_factor} for latent")

    return nnet, nnet_ema, AE


def get_model(config):
    from libs.ldm.util import instantiate_from_config

    config.model.params.first_stage_config.target = 'libs.autoencoder.AutoencoderKL'

    AE = instantiate_from_config(config.model.params.first_stage_config).cuda()
    for param in AE.parameters():
        param.requires_grad = False

    if cond: config.model.params.unet_config.params.num_classes = config.clsnum + 1
    else: config.model.params.unet_config.params.num_classes = None

    unet_config = config.model.params.unet_config
    ae_config = config.model.params.first_stage_config

    unet_config.params.in_channels = ae_config.params.embed_dim
    unet_config.params.out_channels = ae_config.params.embed_dim

    unet_config.params.context_dim = None
    unet_config.params.use_spatial_transformer = False

    nnet = instantiate_from_config(config.model.params.unet_config).cuda()
    nnet_ema = LitEma(nnet)
    print(f"Keeping EMAs of {len(list(nnet_ema.buffers()))}.")

    AE.eval()
    nnet.eval()
    nnet_ema.eval()
    return nnet, nnet_ema, AE


def get_data_loader(ldf_name, udf_name):
    df = pd.read_csv(ldf_name)
    df = df[df['split'] == 'train'].reset_index()
    dataset = CifarDataset(df=df)
    labeled_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=8)

    df = (pd.read_csv(udf_name))
    df = df[df['split'] == 'unlabeled'].reset_index()
    dataset = CifarDataset(df=df)
    unlabeled_loader = DataLoader(dataset, batch_size=300, shuffle=True, num_workers=8)

    return labeled_loader, unlabeled_loader


def main(config):
    ldf_name = './data/Cifars_100labels_path.csv'
    udf_name = './data/Cifars_100labels_path.csv'
    labeled_loader, unlabeled_loader = get_data_loader(ldf_name, udf_name)

    # =============================================================================================================== #

    # model_path = Path(config.prefix, 'Exp_name', 'ckpt', 'model checkpoint name.pt')
    model_path = Path(config.prefix, 'DWD_Reproducing_Joint', 'ckpt', 'DWD_Reproducing_Joint_best.pt')
    assert model_path.exists()

    fname = 'Cifars'
    root = './Generated'
    # =============================================================================================================== #

    # Initialize model and Load checkpoint

    nnet, nnet_ema, AE = get_model(config)
    nnet, nnet_ema, AE = load_checkpoint(nnet, nnet_ema, AE, model_path, (labeled_loader, unlabeled_loader))

    reps = get_labeled_latent(AE, labeled_loader, config)
    cls_num = config.clsnum
    # ================================================================================================================ #

    def cfg_nnet(x, timesteps, c=None, get_rep=False):
        with nnet_ema.ema_scope(nnet):
            # Conditional
            kwargs = {'y': c}
            _cond = nnet(x, timesteps, **kwargs)

            # Unconditional
            kwargs = {'y': torch.tensor([cls_num] * c.size(0)).cuda()}
            _uncond = nnet(x, timesteps, **kwargs)
        return _cond + config.w * (_cond - _uncond)

    def sample_fn(noised_inputs, labs=None, depth=None):
        cond_key = 'c' if config.cfg else 'y'

        if cond:  # Conditional
            kwargs = {cond_key: labs}
        else:  # Unconditional
            kwargs = {cond_key: torch.ones_like(labs) * 10}

        noise_schedule = NoiseScheduleVP(schedule='linear')
        model_fn = model_wrapper(score_model.noise_pred, noise_schedule, time_input_type='0', model_kwargs=kwargs)
        dpm_solver = DPM_Solver(model_fn, noise_schedule)

        z = dpm_solver.sample(noised_inputs, steps=10, eps=1e-3, order=1, T=depth, adaptive_step_size=False, fast_version=True,)
        z = 1. / AE.scale_factor * z
        return z

    # ================================================================================================================ #

    if config.cfg: score_model = sde.ScoreModel(cfg_nnet, pred='noise_pred', sde=sde.VPSDE())
    else: score_model = sde.ScoreModel(nnet, pred='noise_pred', sde=sde.VPSDE())

    depths = [30]
    for depth in depths:
        true_depth = int(1000 // 50 * depth) if isinstance(depth, int) else 999
        # true_depth = 999
        print("\n=> Current depth:", depth)

        if cond:
            if config.pseudo:
                fname += f'_Pseudo_{depth}depth'

            else: fname += f'_Random_{depth}depth'
        else: fname += f'_Uncond_{depth}depth'

        prefix = Path(root, fname)
        prefix.mkdir(parents=True, exist_ok=True)
        if prefix.exists(): print("\n==> Generated samples would be save in EXISTING a folder:", prefix)
        else: print("\n==> CREATE a folder to save the generated samples at:", prefix)

        # =========================================================================================================== #
        new_path, new_label = [], []
        total, n_gen = 0, 0

        pbar = tqdm(total=len(unlabeled_loader))
        print("\n==> Total Iteration:", len(unlabeled_loader))
        for index, (inputs_u, _) in enumerate(unlabeled_loader):
            inputs_u = inputs_u.cuda()

            z = AE.encode(inputs_u).sample()
            z, labs = get_pseudo_label(AE, z, config=config, supports=reps)

            t = (torch.ones(len(z)) * torch.tensor(depth / 50)).cuda()
            noised_z = score_model.sde.xt_sample(z, t)
            samples = unpreprocess(AE.decode(sample_fn(noised_z, labs, depth=(true_depth / total_steps))))

            for idx, gen_u in enumerate(samples):
                if config.pseudo: gen_path = os.path.join(prefix, 'gen_{}_{}class_{}.png'.format(total, labs[idx].item(), Cifar10_label[labs[idx].item()]))
                else: gen_path = os.path.join(prefix, 'gen_{}.png'.format(total))
                save_image(gen_u, gen_path)

                total += 1
                new_path.append(gen_path)
                new_label.append(labs[idx].item())

            n_gen += len(samples)
            pbar.set_description(
                f"Iter: [{index+1:>4d}/{len(unlabeled_loader):>4d}]  "
                f"# of samples: {n_gen}/{60000}."
            )
            pbar.update()

        cifars_df = pd.read_csv(ldf_name)

        paths = list(cifars_df.iloc[cifars_df[cifars_df['split'] != 'unlabeled'].index]['path'])
        labels = list(cifars_df.iloc[cifars_df[cifars_df['split'] != 'unlabeled'].index]['label'])
        splits = list(cifars_df.iloc[cifars_df[cifars_df['split'] != 'unlabeled'].index]['split'])

        paths.extend(new_path)
        labels.extend(new_label)
        splits.extend(['unlabeled' for _ in new_path])

        data = dict(path=paths, label=labels, split=splits)
        new_df = pd.DataFrame(data=data, index=range(len(paths)))

        df_path = f'./data/Cifars_100labels_gen_{fname}.csv'
        new_df.to_csv(df_path)
        print(f"CSV file saved at {df_path}")

    print("=" * 100)
    print("...DONE")


if __name__ == '__main__':
    config, ae_config = set_config(args)
    cond = (config.cond != 'uncond')

    main(config)
