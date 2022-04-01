# Self Supervised Learning with Fastai
> Implementation of popular SOTA self-supervised learning algorithms as Fastai Callbacks.


![CI](https://github.com/KeremTurgutlu/self_supervised/actions/workflows/main.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/self-supervised?color=blue&label=pypi%20version)](https://pypi.org/project/self-supervised/#description)
[![DOI](https://zenodo.org/badge/295835009.svg)](https://zenodo.org/badge/latestdoi/295835009) 

## Install

`pip install self-supervised`

## Documentation

Please read the documentation [here](https://keremturgutlu.github.io/self_supervised).

To go back to github repo please click [here](https://github.com/keremturgutlu/self_supervised/tree/master/).

## Algorithms

Please read the papers or blog posts before getting started with an algorithm, you may also check out documentation page of each algorithm to get a better understanding.

Here are the list of implemented **self_supervised.vision** algorithms:

- [SimCLR v1](https://arxiv.org/pdf/2002.05709.pdf) & [SimCLR v2](https://arxiv.org/pdf/2006.10029.pdf) 
- [MoCo v1](https://arxiv.org/pdf/1911.05722.pdf) & [MoCo v2](https://arxiv.org/pdf/2003.04297.pdf)
- [BYOL](https://arxiv.org/pdf/2006.07733.pdf)
- [SwAV](https://arxiv.org/pdf/2006.09882.pdf)
- [Barlow Twins](https://arxiv.org/pdf/2103.03230.pdf)
- [DINO](https://arxiv.org/pdf/2104.14294.pdf)
- [SupCon](https://arxiv.org/pdf/2004.11362.pdf)

Here are the list of implemented **self_supervised.multimodal** algorithms:

- [CLIP](https://arxiv.org/pdf/2103.00020.pdf)
- CLIP-MoCo (No paper, own idea)

For vision algorithms all models from [timm](https://github.com/rwightman/pytorch-image-models) and [fastai](https://github.com/fastai/fastai) can be used as encoders.

For multimodal training currently CLIP supports ViT-B/32 and ViT-L/14, following best architectures from the paper.

## Simple Usage

### Vision

#### SimCLR

```python
from self_supervised.vision.simclr import *
dls = get_dls(resize, bs)
# encoder = create_encoder("xresnet34", n_in=3, pretrained=False) # a fastai encoder
encoder = create_encoder("tf_efficientnet_b4_ns", n_in=3, pretrained=False) # a timm encoder
model = create_simclr_model(encoder, hidden_size=2048, projection_size=128)
aug_pipelines = get_simclr_aug_pipelines(size=size)
learn = Learner(dls,model,cbs=[SimCLR(aug_pipelines, temp=0.07)])
learn.fit_flat_cos(100, 1e-2)
```

#### MoCo

```python
from self_supervised.vision.moco import *
dls = get_dls(resize, bs)
# encoder = create_encoder("xresnet34", n_in=3, pretrained=False) # a fastai encoder
encoder = create_encoder("tf_efficientnet_b4_ns", n_in=3, pretrained=False) # a timm encoder
model = create_moco_model(encoder, hidden_size=2048, projection_size=128)
aug_pipelines = get_moco_aug_pipelines(size=size)
learn = Learner(dls, model,cbs=[MOCO(aug_pipelines=aug_pipelines, K=128)])
learn.fit_flat_cos(100, 1e-2)
```

#### BYOL

```python
from self_supervised.vision.byol import *
dls = get_dls(resize, bs)
# encoder = create_encoder("xresnet34", n_in=3, pretrained=False) # a fastai encoder
encoder = create_encoder("tf_efficientnet_b4_ns", n_in=3, pretrained=False) # a timm encoder
model = create_byol_model(encoder, hidden_size=2048, projection_size=128)
aug_pipelines = get_byol_aug_pipelines(size=size)
learn = Learner(dls, model,cbs=[BYOL(aug_pipelines=aug_pipelines)])
learn.fit_flat_cos(100, 1e-2)
```

#### SWAV 

```python
from self_supervised.vision.swav import *
dls = get_dls(resize, bs)
encoder = create_encoder("xresnet34", n_in=3, pretrained=False) # a fastai encoder
encoder = create_encoder("tf_efficientnet_b4_ns", n_in=3, pretrained=False) # a timm encoder
model = create_swav_model(encoder, hidden_size=2048, projection_size=128)
aug_pipelines = get_swav_aug_pipelines(num_crops=[2,6],
                                       crop_sizes=[128,96], 
                                       min_scales=[0.25,0.05],
                                       max_scales=[1.0,0.3])
learn = Learner(dls, model, cbs=[SWAV(aug_pipelines=aug_pipelines, crop_assgn_ids=[0,1], K=bs*2**6, queue_start_pct=0.5)])
learn.fit_flat_cos(100, 1e-2)
```

#### Barlow Twins

```python
from self_supervised.vision.barlow_twins import *
dls = get_dls(resize, bs)
# encoder = create_encoder("xresnet34", n_in=3, pretrained=False) # a fastai encoder
encoder = create_encoder("tf_efficientnet_b4_ns", n_in=3, pretrained=False) # a timm encoder
model = create_barlow_twins_model(encoder, hidden_size=2048, projection_size=128)
aug_pipelines = get_barlow_twins_aug_pipelines(size=size)
learn = Learner(dls,model,cbs=[BarlowTwins(aug_pipelines, lmb=5e-3)])
learn.fit_flat_cos(100, 1e-2)
```

#### DINO

```python
from self_supervised.models.vision_transformer import *
from self_supervised.vision.dino import *
dls = get_dls(resize, bs)

deits16 = MultiCropWrapper(deit_small(patch_size=16, drop_path_rate=0.1))
dino_head = DINOHead(deits16.encoder.embed_dim, 2**16, norm_last_layer=True)
student_model = nn.Sequential(deits16,dino_head)

deits16 = MultiCropWrapper(deit_small(patch_size=16))
dino_head = DINOHead(deits16.encoder.embed_dim, 2**16, norm_last_layer=True)
teacher_model = nn.Sequential(deits16,dino_head)

dino_model = DINOModel(student_model, teacher_model)
aug_pipelines = get_dino_aug_pipelines(num_crops=[2,6],
                                       crop_sizes=[128,96], 
                                       min_scales=[0.25,0.05],
                                       max_scales=[1.0,0.3])
 learn = Learner(dls,model,cbs=[DINO(aug_pipelines=aug_pipelines)])
learn.fit_flat_cos(100, 1e-2)
```

#### SupCon

```python
from self_supervised.vision.supcon import *
dls = get_dls(resize, bs)
# encoder = create_encoder("xresnet34", n_in=3, pretrained=False) # a fastai encoder
encoder = create_encoder("tf_efficientnet_b4_ns", n_in=3, pretrained=False) # a timm encoder
model = create_supcon_model(encoder, hidden_size=2048, projection_size=128)
aug_pipelines = get_supcon_aug_pipelines(size=size)
learn = Learner(dls,model,cbs=[SupCon(aug_pipelines, unsup_class_id, unsup_method=UnsupMethod.All,                                               reg_lambda=1.0, temp=0.07)])
learn.fit_flat_cos(100, 1e-2)
```

### Multimodal

#### CLIP

```python
from self_supervised.multimodal.clip import *
dls = get_dls(...)
clip_tokenizer = ClipTokenizer()
vitb32_config_dict = vitb32_config(224, clip_tokenizer.context_length, clip_tokenizer.vocab_size)
clip_model = CLIP(**vitb32_config_dict, checkpoint=False, checkpoint_nchunks=0)
learner = Learner(dls, clip_model, loss_func=noop, cbs=[CLIPTrainer()])
learn.fit_flat_cos(100, 1e-2)
```

#### CLIP-MoCo

```python
from self_supervised.multimodal.clip_moco import *
dls = get_dls(...)
clip_tokenizer = ClipTokenizer()
vitb32_config_dict = vitb32_config(224, clip_tokenizer.context_length, clip_tokenizer.vocab_size)
clip_model = CLIPMOCO(K=4096,m=0.999, **vitb32_config_dict, checkpoint=False, checkpoint_nchunks=0)
learner = Learner(dls, clip_model, loss_func=noop, cbs=[CLIPMOCOTrainer()])
learn.fit_flat_cos(100, 1e-2)
```

## ImageWang Benchmarks

All of the algorithms implemented in this library have been evaluated in [ImageWang Leaderboard](https://github.com/fastai/imagenette#image%E7%BD%91-leaderboard). 

In overall superiority of the algorithms are as follows `SwAV > MoCo > BYOL > SimCLR` in most of the benchmarks. For details you may inspect the history of [ImageWang Leaderboard](https://github.com/fastai/imagenette#image%E7%BD%91-leaderboard) through github. 

`BarlowTwins` is still under testing on ImageWang.

It should be noted that during these experiments no hyperparameter selection/tuning was made beyond using `learn.lr_find()` or making 
sanity checks over data augmentations by visualizing batches. So, there is still space for improvement and overall rankings of the alogrithms may change based on your setup. Yet, the overall rankings are on par with the papers.



## Contributing

Contributions and or requests for new self-supervised algorithms are welcome. This repo will try to keep itself up-to-date with recent SOTA self-supervised algorithms.

Before raising a PR please create a new branch with name `<self-supervised-algorithm>`. You may refer to previous notebooks before implementing your Callback.

Please refer to sections `Developers Guide, Abbreviations Guide, and Style Guide` from https://docs.fast.ai/dev-setup and note that same rules apply for this library.
