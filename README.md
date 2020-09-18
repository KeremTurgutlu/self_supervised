# Self Supervised Learning Fastai Extension
> Implementation of popular SOTA self-supervised learning algorithms as Fastai Callbacks.


You may find documentation [here](https://keremturgutlu.github.io/self_supervised)

## Install

`pip install <>`

## Algorithms

Here are the list of implemented algorithms:

- [SimCLR](https://arxiv.org/pdf/2002.05709.pdf)
- [BYOL](https://arxiv.org/pdf/2006.07733.pdf)
- [SwAV](https://arxiv.org/pdf/2006.09882.pdf)

### ImageWang Benchmarks

All of the algorithms implemented in this library have been evaluated in [ImageWang Leaderboard](https://github.com/fastai/imagenette#image%E7%BD%91-leaderboard). 

In overall superiority of the algorithms are as follows `SwAV > BYOL > SimCLR` in most of the benchmarks. For details you may inspect the history of [ImageWang Leaderboard](https://github.com/fastai/imagenette#image%E7%BD%91-leaderboard) through github. 

## Contributions

Contributions and or requests for new self-supervised algorithms are welcome. This repo will try to keep itself up-to-date with recent SOTA self-supervised algorithms.

Before raising a PR please create a new branch with name `<self-supervised-algorithm>`. You may refer to previous notebooks before implementing your Callback.
