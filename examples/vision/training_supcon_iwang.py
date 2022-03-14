from fastai.vision.all import *
torch.backends.cudnn.benchmark = True
from fastai.callback.wandb import WandbCallback
import wandb
from self_supervised.augmentations import *
from self_supervised.layers import *
from self_supervised.vision.supcon import *
from self_supervised.vision.metrics import *
import timm


def get_dls(size, bs, n_per_class=30, workers=None):
    path = URLs.IMAGEWANG_160 if size <= 160 else URLs.IMAGEWANG
    source = untar_data(path)
    files = get_image_files(source, folders=['unsup', 'train'])

    
    labels = [o.parent.name for i,o in enumerate(files)]
    split_df = pd.DataFrame(labels, columns=['label']).reset_index()
    valid_idxs = split_df.query("label != 'unsup'").groupby("label").sample(n_per_class)['index'].values
    split_df['is_valid'] = False
    split_df.loc[split_df['index'].isin(valid_idxs), 'is_valid'] = True
    train_idxs = split_df[~split_df.is_valid]['index'].values
    valid_idxs = split_df[split_df.is_valid]['index'].values

    
    tfms = [[PILImage.create, ToTensor, RandomResizedCrop(size, min_scale=1.)], 
            [parent_label, Categorize()]]
    dsets = Datasets(files, tfms=tfms, splits=[train_idxs, valid_idxs])
    batch_tfms = [IntToFloatTensor]
    dls = dsets.dataloaders(bs=bs, num_workers=workers, after_batch=batch_tfms)
    return dls


default_configs = dict(
    arch = "xresnet34", # ["xresnet34", "resnet34d"]
    lr = 1e-2,
    wd = 1e-2,
    opt_func = "adam", # ["adam", "lamb"]
    reg_lambda = 1.,
    temp = 0.1,
    unsup_method = "all" # ["all", "only"]
    )

resize, size = 256, 224

default_configs["Resize"] = resize
default_configs["Size"] = size
default_configs["Algorithm"] = "SupCon"

wandb.init(project="self-supervised-imagewang", config=default_configs)
config = wandb.config

if config.arch == "xresnet34":  
    bs = 320
    encoder = create_encoder(config.arch, pretrained=False, n_in=3)
elif config.arch == "resnet34d":
    bs = 360
    encoder = CheckpointResNet(create_encoder(config.arch, pretrained=False, n_in=3), checkpoint_nchunks=2)

dls = get_dls(resize, bs)
model = create_supcon_model(encoder)
aug_pipelines = get_supcon_aug_pipelines(size, rotate=True, rotate_deg=10, jitter=True, bw=True, blur=False) 
cbs=[SupCon(aug_pipelines, 
            unsup_class_id = dls.vocab.o2i['unsup'], 
            unsup_method = config.unsup_method, 
            reg_lambda = config.reg_lambda, 
            temp = config.temp)]
cbs += [WandbCallback(log_preds=False, log_model=False)]
cbs += [TerminateOnNaNCallback()]


knn_metric_cb = KNNProxyMetric()
cbs += [knn_metric_cb]
metric = ValueMetric(knn_metric_cb.accuracy, metric_name='knn_accuracy')

if config.opt_func == "adam":
    opt_func = Adam
elif config.opt_func == "lamb":
    opt_func = Lamb

learn = Learner(dls, model, opt_func=opt_func, cbs=cbs, metrics=metric)
learn.to_fp16()

learn.unfreeze()
learn.fit_flat_cos(100, config.lr, wd=config.wd, pct_start=0.5)


save_name = f'{wandb.run.name}'
learn.save(save_name)

if config.arch == "xresnet34":  
    torch.save(learn.model.encoder.state_dict(), learn.path/learn.model_dir/f'{save_name}_encoder.pth')
elif config.arch == "resnet34d":
    torch.save(learn.model.encoder.resnet_model.state_dict(), learn.path/learn.model_dir/f'{save_name}_encoder.pth')


wandb.finish()
