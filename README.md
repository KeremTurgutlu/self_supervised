# Self Supervised Learning Fastai Extension
> Implementation of popular SOTA self-supervised learning algorithms as Fastai Callbacks.


You may find documentation [here](https://keremturgutlu.github.io/self_supervised) and github repo [here](https://github.com/keremturgutlu/self_supervised/tree/master/)

## Install

`pip install self-supervised`

## Algorithms

Here are the list of implemented **self_supervised.vision** algorithms:

- [SimCLR]()
- [MoCo]()
- [BYOL]()
- [SwAV]()

Here are the list of implemented **self_supervised.multimodal** algorithms:

- [CLIP]()
- [CLIP-MoCo]() (No paper, own idea)

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
from self_supervised.vision.moco import **
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

It should be noted that during these experiments no hyperparameter selection/tuning was made beyond using `learn.lr_find()` or making sanity checks over data augmentations by visualizing batches. So, there is still space for improvement and overall rankings of the alogrithms may change based on your setup. Yet, the overall rankings are on par with the papers.

## Contributing

Contributions and or requests for new self-supervised algorithms are welcome. This repo will try to keep itself up-to-date with recent SOTA self-supervised algorithms.

Before raising a PR please create a new branch with name `<self-supervised-algorithm>`. You may refer to previous notebooks before implementing your Callback.

Please refer to sections `Developers Guide, Abbreviations Guide, and Style Guide` from https://docs.fast.ai/dev-setup and note that same rules apply for this library.
