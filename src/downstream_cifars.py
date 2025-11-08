import copy
import math
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import torchvision.models.vgg
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import classification_report
from tools.utils import *
from tools.randaugment import RandAugmentMC
from PIL import Image as im
import random


parser = argparse.ArgumentParser(description='PyTorch Downstream Training')
parser.add_argument('--init', default='Random', type=str, metavar='Model Init method', choices=['Self', 'Semi', 'ImageNet', 'Random'],
                    dest='init')
parser.add_argument('--epochs', default=256, type=int, metavar='Maximum epochs', help='Maximum epochs', dest='epochs')
parser.add_argument('--layer', default='linear', type=str, metavar='Additional Layer', help='linear|2layer',
                    dest='layer')
parser.add_argument('--trainable', default=True, type=lambda s: s in ['True', 'true', 1], metavar='Trainable',
                    help='fine-tune / linear-eval', dest='trainable')
parser.add_argument('--arch', default='wideresnet', type=str, metavar='Base Encoder Architecture',
                    help='DenseNet|ResNet', dest='arch')
parser.add_argument('--lr', default=0.1, type=float, metavar='Learning rate',
                    help='Learning rate', dest='lr')
parser.add_argument('--cls', default=True, type=lambda s: s in ['True', 'true', 1], metavar='classifier init',
                    help='classifier init', dest='cls')
parser.add_argument('--indicator', default='Acc', type=str, metavar='Metric Indicator', help='AUC|ACC',
                    dest='indicator')
parser.add_argument('--cos', default=1, type=int, metavar='Cosine Decay Rate', help='cosine decay rate',
                    dest='cos_rate')
parser.add_argument('--lr-decay', default=True, type=lambda s: s in ['True', 'true', 1], metavar='lr decay',
                    help='learning rate decay while training', dest='decay')
parser.add_argument('--lr-scheduler', default='cosine', type=str, metavar='lr scheduler',
                    help='ReduceLROnPlateau|Cosine', dest='scheduler')
parser.add_argument('--task', default='cifars', type=str, metavar='Target task', choices=['six', 'cifars', 'imagenet', 'inat'], dest='task')
parser.add_argument('--source', default='PAWS', type=str, metavar='Pretrained Source', help='STL|Tiny|PAWS', dest='source')
parser.add_argument('--optim', default='AdamW', type=str, metavar='Optimizer', help='SGD|Adam', dest='optim')
parser.add_argument('--bs', default=64, type=int, metavar='Batch size', help='batch size', dest='batch_size')
parser.add_argument('--wd', default=1e-4, type=float, metavar='Weight Decay', help='weight decay value for SGD', dest='wd')
parser.add_argument('--nest', default=True, type=lambda s: s in ['True', 'true', 1], metavar='nesterov option for SGD', help='nesterov option for SGD', dest='nesterov')
parser.add_argument('--frac', default=1.0, type=float, dest='frac')
parser.add_argument('--exp_name', default=None, type=str, required=False, dest='exp_name')
parser.add_argument('--wandb', default=False, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--num_labels', type=str, default=100)
parser.add_argument('--prefix', default='./assets/SL', type=str, dest='prefix')

parser.add_argument('--num_classes', type=str, default=10)
parser.add_argument('--steps', type=int, default=1024)
parser.add_argument('--mu', type=int, default=1)
parser.add_argument('--rand_aug', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--label-smoothing', default=0.15, type=float, help='label smoothing alpha')
parser.add_argument('--seed', default=None, type=int)

parser.add_argument('--amp', default=True, type=lambda s: s in ['True', 'true', 1])
parser.add_argument('--scaler', default=True, type=lambda s: s in ['True', 'true', 1])

Cifar_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CifarDataset(Dataset):
    def __init__(self, data_list, config, mode='train'):
        self.config = config
        self.data_list = data_list
        self.df = self.config.df
        self.mode = mode
        self.cropped_size = 32

        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        img_size = 32

        if mode == 'train':
            if config.rand_aug:
                self.transpose = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=img_size,
                                          padding=int(img_size * 0.125),
                                          padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ])
            else:
                self.transpose = transforms.Compose([
                    transforms.RandomCrop(size=img_size,
                                          padding=int(img_size * 0.125),
                                          padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ])

        elif mode == 'unlabeled':
            if config.rand_aug:
                self.transpose = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=img_size,
                                          padding=int(img_size * 0.125),
                                          padding_mode='reflect'),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ])
            else:
                self.transpose = transforms.Compose([
                    transforms.RandomCrop(size=img_size,
                                          padding=int(img_size * 0.125),
                                          padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD),
                ])
        else:
            self.transpose = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        index = self.data_list[idx]
        contents = self.df.loc[index]

        path = contents['path']
        img = im.open(path).convert('RGB')
        label = contents['label']

        x = self.transpose(img)
        y = torch.tensor(label)

        return x, y


class Config(object):
    def __init__(self):
        self.epochs = 100
        self.start_epoch = 0
        self.batch_size = 256
        self.learning_rate = 0.05
        self.df = None
        self.resume = False
        self.ratio = 64
        self.aug_plus = False
        self.value = AverageMeter('Value', ':6.2f')
        self.epoch = 0

    def call(self, *args, **kwargs):
        return self


def set_seed(seed):
    if seed is None:
        seed = np.random.randint(0, 1000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resuming(model):
    if os.path.isfile(config.resume):
        checkpoint = torch.load(config.resume)
        model.load_state_dict(checkpoint['state_dict'])
        config.start_epochs = checkpoint['epoch']
        print("==> load checkpoint from '{}' (epoch {})".format(config.resume, checkpoint['epoch']))
    else:
        print("==> no checkpoint found at '{}'".format(config.resume))
    return model


def set_config(config=None, args=None):
    config = Config()

    if args is not None:
        keys = list(args.__dict__.keys())
        values = list(args.__dict__.values())
        [setattr(config, keys[i], values[i]) for i in range(len(keys))]

    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device: {}".format(config.device))

    return config


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


def get_train_setting(model, config):
    params = model.parameters()
    # params = model.net.parameters()

    if config.optim == 'Adam': optimizer = optim.Adam(params, lr=config.lr)
    elif config.optim == 'AdamW': optimizer = optim.AdamW(params, lr=config.lr)
    else: optimizer = optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=0, nesterov=config.nesterov)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()

    if config.scheduler == 'cosine': lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    else: lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.0001, min_lr=0, verbose=True)

    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(save_path, "{}.pt".format(config.file_name)), early_count_verbose=False)
    scaler = GradScaler()

    return optimizer, criterion, lr_scheduler, early_stopping, scaler


def get_dataset(config):
    config.target_task = 'downstream'
    task = config.task.lower()

    # df_name = 'Cifars_100labels_path'
    df_name = 'Cifars_100labels_gen_Cifars_Pseudo_30depth'
    Dataset = CifarDataset

    config.df = pd.read_csv(f'./data/{df_name}.csv')

    train_list = list(config.df[config.df['split'] == 'train'].index)
    val_list = list(config.df[config.df['split'] == 'test'].index)
    test_list = list(config.df[config.df['split'] == 'test'].index)
    unlabeled_list = list(config.df[config.df['split'] == 'unlabeled'].index)

    train_gen = Dataset(train_list, config)
    val_gen = Dataset(val_list, config, mode='validation')
    test_gen = Dataset(test_list, config, mode='test')
    unlabeled_gen = Dataset(unlabeled_list, config, mode='unlabeled')

    train_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_gen, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_gen, batch_size=args.batch_size, shuffle=False, num_workers=4)
    unlabeled_loader = DataLoader(unlabeled_gen, batch_size=args.batch_size * args.mu, shuffle=True, num_workers=4)

    return train_loader, valid_loader, test_loader, unlabeled_loader


def construct_model(arch=None, classifier='linear', load_path: str = None, trainable=True, pretrained=False, config=None):
    if 'vgg' in arch:
        model_ = torchvision.models.vgg.vgg16(pretrained=True)
        model_.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(p=0.5), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(p=0.5), nn.Linear(4096, len(target_label)))
        model_.name = 'VGG16'
        return model_.cuda()

    if 'wideresnet' in arch:
        import tools.wideresnet as models
        model_ = models.build_wideresnet(depth=28, widen_factor=2, dropout=0, num_classes=len(target_label))
        model_.name = 'WideResnet'
        return model_.cuda()

    base_mode = torchvision.models.__dict__[arch]
    model_ = LinearWrapper(base_encoder=base_mode, load_path=load_path, trainable=trainable, pretrained=pretrained, num_classes=len(target_label), config=config)
    return model_.cuda()


# -----------------------------------------------------------------------------------------------------------------------------------------
def MC_evaluation(model, data_loader, criterion):
    model.eval()

    eval_loss, total, correct = 0.0, 0, 0
    output, label = torch.FloatTensor().cuda(),  torch.FloatTensor().cuda()

    with torch.no_grad():
        for idx, (x, y) in enumerate(data_loader):
            x, y = x.cuda(), y.to(torch.int64).cuda()

            y_pred = model(x)

            loss = criterion(y_pred, y)
            eval_loss += loss.item()

            output = torch.cat((output, y_pred.detach()), 0)
            label = torch.cat((label, y.detach()), 0)

            _, predicted = y_pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    eval_loss = eval_loss / (idx+1)
    eval_acc = correct / total * 100.

    return eval_loss, eval_acc


def MC_train(model, data_generator, optimizer, criterion, config):
    losses = AverageMeter("Loss", ":>5.4f")
    accuracies = AverageMeter('ACC', ":>5.4f")
    model.train()

    total, correct = 0, 0
    output, label = torch.FloatTensor().cuda(), torch.FloatTensor().cuda()
    end = time.time()

    pbar = tqdm(total=args.steps)
    for batch_idx in range(args.steps):
        batch = next(data_generator)
        x, y = batch['image'].cuda(), batch['class_label'].to(torch.int64).cuda()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        losses.update(loss.item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = torch.cat((output, y_pred.detach()), 0)
        label = torch.cat((label, y.detach()), 0)

        _, predicted = y_pred.max(1)
        labels = y.to(config.device)
        total += y.size(0)
        correct += predicted.eq(labels).sum().item()

        step_acc = correct / total * 100.
        accuracies.update(step_acc, x.size(0))

        txt = (f"Epoch: [{config.epoch:>4d}/{config.epochs:>4d}] "
               f"loss: {losses.avg:>.5f}. acc: {accuracies.avg:>4f}. ")
        pbar.set_description(txt)
        pbar.update()

    return losses.get_avg(), accuracies.get_avg()


# -----------------------------------------------------------------------------------------------------------------------------------------
def main(config):
    print("==> construct model start: {}".format(config.arch))
    model = construct_model(config.arch, config.layer, config)
    config.file_name = '{}_{}'.format(task.upper(), args.arch)
    print("...DONE\n")

    print("==> training settings start")
    optimizer, criterion, lr_scheduler, early_stopping, scaler = get_train_setting(model, config)

    train_loader, valid_loader, test_loader, unlabeled_loader = get_dataset(config)
    data_generator = get_data_generator(labeled_loader=train_loader, unlabeled_loader=unlabeled_loader, config=config)
    print("...DONE\n")

    print("==> classifier: {}".format(model.name))
    print("==> file name: {}".format(config.file_name))
    print("{}\n".format("=" * 100))

    if config.resume: model = resuming(model)

    train = MC_train
    evaluation = MC_evaluation

    state_dict = model.state_dict()
    min_loss, max_acc = evaluation(model, valid_loader, criterion)
    print("Initial Loss: {:.4f}, {}: {:.4f}%".format(min_loss, config.indicator, max_acc))

    for epoch in range(config.epochs):
        config.epoch = epoch + 1
        start_time = time.time()
        print("\nCurrent epoch: {}, learning rate: {:.8f}".format(config.epoch, optimizer.param_groups[0]['lr']))

        train_loss, train_acc = train(model, data_generator, optimizer, criterion, config)
        valid_loss, valid_acc = evaluation(model, valid_loader, criterion)

        print("\nEpoch: {}  Time taken: {}s  Valid Loss: {:.4f}  Valid {}: {:.4f}% ".format(config.epoch, np.round((time.time() - start_time)), valid_loss, config.indicator, valid_acc))
        early_stop = early_stopping(valid_loss, model)

        if valid_acc > max_acc:
            max_acc = valid_acc
            state_dict = model.state_dict()

        if args.decay:
            if config.scheduler == 'cosine':
                lr_scheduler.step()
            else:
                lr_scheduler.step(valid_loss, epoch)

        print("=" * 100)
        print("Best ACC:", np.round(max_acc, 4))
        print()

    model.load_state_dict(state_dict)
    train_loss, train_acc = evaluation(model, train_loader, criterion)
    test_loss, test_acc = evaluation(model, test_loader, criterion)

    print("Best Val ACC:", np.round(max_acc, 4))
    print("[Train] Loss: {:.4f}  ACC: {:.4f}".format(train_loss, train_acc))
    print("[Test] Loss: {:.4f}  ACC: {:.4f}".format(test_loss, test_acc))

    print("...DONE")


# -----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    config = set_config(None, args)

    target_label = Cifar_label
    task = 'Cifars'

    save_path = Path(config.prefix, task, config.exp_name)
    save_path.mkdir(exist_ok=True, parents=True)

    set_seed(config.seed)
    main(config)


