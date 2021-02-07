from fastai.vision.all import *
from self_supervised.augmentations import *
from self_supervised.layers import *
from self_supervised.swav import *


sqrmom=0.99
mom=0.95
beta=0.
eps=1e-4
opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)


def get_dls(size, bs, workers=None):
    path = URLs.IMAGEWANG_160 if size <= 160 else URLs.IMAGEWANG
    source = untar_data(path)
    
    files = get_image_files(source)
    tfms = [[PILImage.create, ToTensor, RandomResizedCrop(size, min_scale=0.9)], 
            [parent_label, Categorize()]]
    
    dsets = Datasets(files, tfms=tfms, splits=RandomSplitter(valid_pct=0.1)(files))
    
    batch_tfms = [IntToFloatTensor]
    dls = dsets.dataloaders(bs=bs, num_workers=workers, after_batch=batch_tfms)
    return dls


bs=128
resize, size = 160, 128


arch_name = "tf_efficientnet_b0_ns"

dls = get_dls(resize, bs)
encoder = create_timm_encoder(arch_name, n_in=3, pretrained=False, pool_type=None)
model = create_swav_model(encoder, n_in=3)
learn = Learner(dls, model, SWAVLoss(),
                cbs=[SWAV(aug_func=get_batch_augs,
                          crop_sizes=[size,96], 
                          num_crops=[2,6],
                          min_scales=[0.25,0.2],
                          max_scales=[1.0,0.35],
                          rotate_deg=10),
                     TerminateOnNaNCallback()])