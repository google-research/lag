# LAG: Latent Adversarial Generator

Code for the paper: 
"[Creating High Resolution Images with a Latent Adversarial Generator](https://arxiv.org/abs/2003.02365)"
by David Berthelot, Peyman Milanfar, and Ian Goodfellow.

This is not an officially supported Google product.


## Setup
**Important**: `ML_DATA` is a shell environment variable that should point to the location where the datasets are
installed. See the *Install datasets* section for more details.

### Install dependencies

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick virtualenv
virtualenv -p python3 --system-site-packages ~/env3
~/env3/bin/pip install -r requirements.txt
```

To use virtualenv, when starting a new shell, always run the following lines first:

```bash
. ~/env3/bin/activate
export PYTHONPATH=$PYTHONPATH:.
```

### Install datasets

#### Directory for storing training data 
```
# You can define ML_DATA to whatever path you like.
export ML_DATA="path to where you want the datasets saved"
mkdir $ML_DATA $ML_DATA/Models
```

Download vgg19.npy from https://github.com/machrisaa/tensorflow-vgg
to `$ML_DATA/Models`.

#### Download datasets

```bash
# Important ML_DATA and PYTONPATH must be defined (see previous sections).
scripts/create_datasets.py
```

Alternatively you can download just some selected datasets to save space and time, for example:
```bash
scripts/create_datasets.py celeba svhn
```

The supported datasets are (`%s` represents the image size, when present it can be replaced by 32, 64, 128 or 256):
```
celeba%s, cifar10, mnist, svhn,

lsun_bedroom%s, lsun_bridge%s, lsun_church_outdoor%s, lsun_classroom%s,
lsun_conference_room%s, lsun_dining_room%s, lsun_kitchen%s, lsun_living_room%s,
lsun_restaurant%s, lsun_tower%s
```

## Running

### Setup

All commands must be ran from the project root. The following environment variables must be defined:
```bash
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

### Example

For example, training `lag` for a 32x zoom using CelebA128:
```bash
CUDA_VISIBLE_DEVICES=0 python lag.py --dataset=celeba128 --scale=32
```


### Multi-GPU training
Just pass more GPUs and the code automatically scales to them, here we assign GPUs 4-7 to the program:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python lag.py --dataset=celeba128 --scale=32
```

### Flags

```bash
python lag.py --help
# The following option might be too slow to be really practical.
# python lag.py --helpfull
# So instead I use this hack to find the flags:
fgrep -R flags.DEFINE libml lag.py
```

### Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:
```bash
tensorboard --logdir experiments
# And point your browser the link printed when tensorboard starts.
```

## Generating samples
Following the previous example, for a LAG model trained on CelebA128, the `ckpt` is the folder where the model was
trained:
```bash
python scripts/lag_generate_candidates.py\
 --dataset=celeba128\
 --samples 0,3,4,6,8,9,12,13,15\
 --ckpt experiments/celeba128/average32X/LAG_batch16_blocks8_filters256_filters_min64_lod_min1_lr0.001_mse_weight10.0_noise_dim64_training_kimg2048_transition_kimg2048_ttur4_wass_target1.0_weight_avg0.999/
```

## Citing this work
```
@article{berthelot2020creating,
  title={Creating High Resolution Images with a Latent Adversarial Generator},
  author={Berthelot, David and Milanfar, Peyman and Goodfellow, Ian},
  journal={arXiv preprint arXiv:2003.02365},
  year={2020}
}
```
