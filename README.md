# DWD
This is the official implementation code for [DWD: Data Augmentation with Diffusion for Open-Set Semi-Supervised Learning](https://openreview.net/pdf?id=OP3sNTIE1O) (NeurIPS 2024')

To reproduce the result, please following the procedure.

### Environment configuration.
- download the assets from [here](https://zenodo.org/records/11246593) and place the files in proper location.
- 'Cifar100.zip, cifar10.zip' locate in '/data' and extract them
- 'Autoencoder_KL_Cifars_16x16x3_Attn.pt' place in '/assets/autoencoder'
- 'train_1000labels.npz' place in 'assets/fid_stats'

### Train DWD
- python train_ldm_DWD_SDE.py --interval --cfg True --rep_with_img True --cond joint --exp_name Reproducing

### Transform the data based on unlabeled data
- python sample_ldm_SDE.py --pseudo True --cond multi


### Train DWD-SL using transformed data
- python downstream_cifars.py --init Random --lr 0.03 --lr-decay True --wd 1e-4 --bs 64 --nest True --optim AdamW --arch wideresnet --epochs 256 --steps 1024 --rand_aug True --label-smoothing 0.15 --seed 74 --exp_name AdamW_003lr_Seed74

### Train FixMatch + DWD-UT using tranformed data
- python train.py --dataset cifar10 --num-labeled 100 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --u_data ood --img_size 32 --exp_name Reproducing


## Qualitative Result

