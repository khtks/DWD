import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
from dotted_dict import DottedDict
import shutil
from datasets import RepDataset, CFGDataset
from torch.utils.data import DataLoader


def snn(query, supports, labels=None, norm=True, squeeze=True, return_sharp=False):
    """ Soft Nearest Neighbours similarity classifier """
    # Step 1: normalize embeddings
    tau, T = 0.1, 0.25
    softmax = torch.nn.Softmax(dim=1)
    gap = torch.nn.AdaptiveAvgPool2d((1, 1))

    if squeeze:
        query = torch.flatten(gap(query), 1)
        supports = torch.flatten(gap(supports), 1)

    if norm:
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)

    if labels is None: labels = torch.tensor(np.arange(0, supports.shape[0]) // 100)

    labels = torch.nn.functional.one_hot(labels, num_classes=10).float().cuda()
    supports, query, labels = supports.detach(), query.detach(), labels.detach()

    # Step 2: compute similarity between local embeddings
    probs = softmax(query @ supports.T / tau)
    soft_p = probs @ labels

    # Step 3: Sharpening the probability
    sharp_p = soft_p ** (1. / T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)

    if return_sharp: return sharp_p
    else: return torch.argmax(sharp_p, dim=1)


@torch.no_grad()
def get_rep_first(LDM, labeled_dataset, unlabeled_dataset, config):
    print("Target: utils.get_rep_first")
    if os.path.exists(config.data.rep_path) and not config.force_save:
        print(f"Load latents from {config.data.rep_path}")
        extracted = torch.load(config.data.rep_path)
        l_reps, l_labs, u_reps, u_labs = extracted['labeled']['rep'], extracted['labeled']['label'], extracted['unlabeled']['rep'], extracted['unlabeled']['label']

        LDM.scale_factor = extracted['scale_factor']
        print(f"At the very first step, setting LDM.scale_factor to {LDM.scale_factor} for pre-extracted latent")

    else:
        LDM.train()
        labeled_loader = DataLoader(labeled_dataset, batch_size=config.lbs, shuffle=True, drop_last=False, num_workers=4,pin_memory=True, persistent_workers=True)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config.ubs, shuffle=True, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True)

        l_reps, l_labs = torch.FloatTensor(), torch.LongTensor()
        u_reps, u_labs = torch.FloatTensor(), torch.LongTensor()

        (lx, _), (ux, _) = next(iter(labeled_loader)), next(iter(unlabeled_loader))
        x = torch.cat((lx, ux), dim=0).cuda()

        encoder_posterior = LDM.encode_first_stage(x)
        z = LDM.get_first_stage_encoding(encoder_posterior).detach()
        del LDM.scale_factor
        LDM.register_buffer('scale_factor', 1. / z.flatten().std())
        print(f"At the very first step, setting LDM.scale_factor to {LDM.scale_factor} for pre-extracted latent")

        print("\n# Extract Labeled data")
        for img, lab in tqdm(labeled_loader):
            encoder_posterior = LDM.encode_first_stage(img.cuda())
            z = LDM.get_first_stage_encoding(encoder_posterior)

            l_reps = torch.cat((l_reps, z.detach().cpu()), dim=0)
            l_labs = torch.cat((l_labs, lab.detach().cpu()), dim=0)

        print("# Extract Unlabeled data")
        for img, lab in tqdm(unlabeled_loader):
            encoder_posterior = LDM.encode_first_stage(img.cuda())
            z = LDM.get_first_stage_encoding(encoder_posterior)

            u_reps = torch.cat((u_reps, z.detach().cpu()), dim=0)
            u_labs = torch.cat((u_labs, lab.detach().cpu()), dim=0)

        torch.save({"labeled": {"rep": l_reps, "label": l_labs}, "unlabeled": {"rep": u_reps, "label": u_labs}, "scale_factor": LDM.scale_factor}, config.data.rep_path)
        print(f"Saving extracted latents on {config.data.rep_path}")

    labeled_dataset = RepDataset(reps=l_reps, labs=l_labs, dataset=list(np.random.permutation(np.arange(l_reps.shape[0]))))
    unlabeled_dataset = RepDataset(reps=u_reps, labs=u_labs, dataset=list(np.random.permutation(np.arange(u_reps.shape[0]))))

    return labeled_dataset, unlabeled_dataset


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def get_lpips(img1, img2, n_samples, img_size):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()

    img1 = img1.reshape(n_samples, 3, img_size, img_size).cuda()
    img2 = img2.reshape(n_samples, 3, img_size, img_size).cuda()

    indices1 = torch.randperm(img1.shape[0])
    indices2 = torch.randperm(img2.shape[0])

    img1 = img1[indices1, :]
    img2 = img1[indices2, :]

    # LPIPS = (lpips(img1, img2) * 0.5 + lpips(img2, img1) * 0.5).mean()
    LPIPS = lpips(img1, img2)

    return LPIPS.item()


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best_{}'.format(filename)))


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'iddpm':
        from libs.iddpm.unet import UNetModel
        return UNetModel(**kwargs)
    elif name == 'adm':
        from libs.guided_diffusion.unet import UNetModel
        return UNetModel(**kwargs)
    elif name == 'uvit':
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    elif name == 'uvit_i2t':
        from libs.uvit_i2t import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t':
        from libs.uvit_t import UViT
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1

    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.model)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.model)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


# ================================================================================================================== #


class ValLogger(object):
    def __int__(self, name):
        self.name = name
        self.loss = []
        self.fid = []
        self.lpips = []

    def update(self, loss, fid, lpips):
        self.loss.append(loss)
        self.fid.append(fid)
        self.lpips.append(lpips)

    def get_best(self):
        return np.min(self.loss), np.min(self.fid), np.max(self.lpips)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', avg_opt=True):
        self.name = name
        self.fmt = fmt
        self.avg_opt = avg_opt
        self.log = []
        self.reset()

    def reset(self):
        # print("in the reset function")
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.log.append(val)

    def __str__(self):
        if self.avg_opt:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        else:
            fmtstr = '{name} {val' + self.fmt + '}'

        fmtstr = fmtstr.strip()

        return fmtstr.format(**self.__dict__)

    def get_avg(self):
        return self.avg

    def get_top(self):
        return np.max(self.log)

    def get_bottom(self):
        return np.min(self.log)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def set_batch_len(self, num_batches):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        return self

    def set_prefix(self, prefix):
        self.prefix = prefix
        return self

    def reset_meters(self):
        list(map(lambda x: x.reset(), self.meters))
        return self


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', best_model=True, save_model=True,
                 early_count_verbose=True, prefix=''):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_model = best_model
        self.early_count_verbose = early_count_verbose
        self.save_model = save_model
        self.prefix = prefix

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            verbose_str = self.prefix + "Validation loss {:.6f} did not decreased from {:.6f}".format(-1 * score, -1 * self.best_score)
            print(verbose_str)
            if self.early_count_verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        if self.early_stop:
            print("Early stopping")

        if self.early_stop and self.best_model:
            return self.early_stop
        else:
            return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            verbose_str = self.prefix + f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            print(verbose_str)
        if self.save_model:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def restore_model(self, model):
        model.load_state_dict(torch.load(self.path))
        return model


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)

# =================================================================================================================== #