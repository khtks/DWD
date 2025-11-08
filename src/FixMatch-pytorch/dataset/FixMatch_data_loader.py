import logging
import math

import numpy as np
import pandas as pd
from PIL import Image as im
from torchvision import datasets
from torchvision import transforms

import torch
from torch.utils.data import Dataset
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

stl_mean = (0.485, 0.456, 0.406)
stl_std = (0.229, 0.224, 0.225)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_cifar10(args):
    img_size = args.img_size
    transform_labeled = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=img_size,
                              padding=int(img_size * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])

    transform_unlabeled = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)])

    u_data = args.u_data
    if u_data == 'ood':
        df_name = f'Cifars_100labels_path'

    elif u_data == 'gen':
        df_name = 'Cifars_100labels_gen_Cifars_Pseudo_30depth'

    args.df_name = df_name
    df = pd.read_csv(f'../data/{df_name}.csv')

    train_list = list(df[df['split'] == 'train'].index)
    val_list = list(df[df['split'] == 'test'].index)
    test_list = list(df[df['split'] == 'test'].index)
    unlabeled_list = list(df[df['split'] == 'unlabeled'].index)

    print(df.tail())
    print(f"\n# of Train: {len(train_list)}  ||  Val {len(val_list)}  ||  Test {len(test_list)}  ||  Unlabeled {len(unlabeled_list)}\n")

    train_labeled_dataset = CIFAR10SSL(df=df, indexs=train_list, labeled=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(df=df, indexs=unlabeled_list, labeled=False, transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std, img_size=args.img_size))
    valid_dataset = CIFAR10SSL(df=df, indexs=val_list, labeled=True, transform=transform_val)
    test_dataset = CIFAR10SSL(df=df, indexs=test_list, labeled=True, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def x_split(args, df):
    label_per_class = args.num_labeled // args.num_classes

    np.random.seed(args.seed)
    data_list = []
    df_ = df[df['split'] == 'train']
    for label in range(0, 10):
        label = float(label)

        labeled_idx = df_[df['label'] == label].index
        ls = np.random.choice(labeled_idx, label_per_class, replace=False)
        data_list.extend(ls)

    print(label_per_class, args.num_labeled, args.num_classes, "||", len(data_list))
    assert len(data_list) == args.num_labeled

    return data_list


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, img_size):
        self.weak = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(Dataset):
    def __init__(self, df, indexs, labeled=True, transform=None, target_transform=None):

        self.df = df
        self.data = indexs
        self.labeled = labeled

        self.transform = transform
        self.target_transform = target_transform

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx = self.data[index]
        contents = self.df.loc[idx]

        path = '../' + contents['path']
        img = im.open(path).convert('RGB')

        target = int(contents['label']) if self.labeled else 0
        target = torch.tensor(target).long()

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.to_tensor(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def DATASET_GETTERS(dname):
    dd = {'cifar10': get_cifar10}
    return dd[dname]
