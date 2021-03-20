# example cmd for distributed training:
# python -m fastai.launch training_clip.py --arch vitb32 --size 224 --bs 360 --epochs 24 --lr 1e-4 --do_finetune True --use_grad_check True --grad_check_nchunks 2

from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import WandbCallback
import wandb
torch.backends.cudnn.benchmark = True

import clip
from self_supervised.multimodal.clip import *
from zero_optimizer import ZeroRedundancyOptimizer


### Dataset
# This section most likely will need modification for your own data

# a CSV file with image content ids, title/text and num_tokens computed by ClipTokenizer()
title_df = pd.read_csv("<<YOUR_DATASET>>.csv")

# Num tokens needs to be <= 77 for validation metric
# If you disable validation metric ignore this part
title_df = title_df.query("num_tokens<=77")

# Extract all content ids
cid2title = dict(zip(title_df['cid'], title_df['title']))
cids = title_df['cid'].values

datapath = Path("<<DIRECTORY_FOR_IMAGES>>")

# Create content id to image path mapping
image_files = get_image_files(datapath)
cid2file = {int(o.stem.split("_")[2]):o for o in image_files}


# content ids, and validation content ids
sample_valid_cids = pd.read_pickle("<<YOUR_VALIDATION_CONTENT_IDS>>>.pkl")
valid_cids = sample_valid_cids[:10000]

def read_image(cid): return PILImage.create(cid2file[cid])
def read_text(cid): return cid2title[cid]
def dummy_targ(o): return 0 # loss func is not called without it


def get_dls(cids,valid_cids,size,bs):
    clip_stats = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    clip_tokenizer = ClipTokenizer()
    
    split_func = lambda cid: True if cid in valid_cids else False
    dsets = Datasets(cids, tfms=[read_image, read_text, dummy_targ], n_inp=2, splits=FuncSplitter(split_func)(cids))
    item_tfms = [RandomResizedCrop(size, min_scale=0.9), clip_tokenizer, ToTensor()]

    batch_tfms = [IntToFloatTensor, Normalize.from_stats(*clip_stats)]
    train_dl = TfmdDL(dsets.train, shuffle=True, bs=bs, after_item=item_tfms, after_batch=batch_tfms, drop_last=True)
    valid_dl = TfmdDL(dsets.valid, shuffle=False, bs=bs*2, after_item=item_tfms, after_batch=batch_tfms)
    dls = DataLoaders(train_dl, valid_dl, device=default_device())
    return dls, clip_tokenizer


# fp16 + grad checkpoint issue fix
# https://github.com/pytorch/pytorch/issues/49738
# https://github.com/pytorch/pytorch/pull/49757/files


@patch
def after_batch(self: WandbCallback):
    "Log hyper-parameters and training loss"
    if self.training:
        self._wandb_step += 1
        self._wandb_epoch += 1/self.n_iter
        hypers = {f'{k}_{i}':v for i,h in enumerate(self.opt.hypers) for k,v in h.items()}
        wandb.log({'epoch': self._wandb_epoch, 
                   'train_loss': self.smooth_loss.clone().detach().cpu(),
                   'raw_loss': self.loss.clone().detach().cpu()},
                    step=self._wandb_step)
    
    

@call_parse
def main(
    arch:               Param("Arch", str)='vitb32',
    size:               Param("Image resolution", int)=224,    
    bs:                 Param("Batch Size", int)=128,
    epochs:             Param("Number of epochs for training", int)=1,    
    lr:                 Param("Learning rate for training", float)=5e-5,
    opt:                Param("Optimizer", str)='zero',
    use_grad_check:     Param("Gradient checkpointing", bool_arg)=True,
    grad_check_nchunks: Param("Number of chunks for gradient checkpoint", int)=2,
    do_finetune:        Param("Whether to do finetuning", bool_arg)=False,
    finetune_modelname: Param("CLIP open source model name to load for finetuning", str)='ViT-B/32'):
    
    WANDB = True
        
    # start wandb
    if rank_distrib() == 0 and WANDB:
        wandb.init(project="XXX", entity="XXX");
        wandb.config.update({"Arch":arch, 
                             "Optimizer": opt,
                             "Size":size,
                             "BS":bs,
                             "Training": "Finetuning" if do_finetune else "From Scratch"});

    # dataloaders
    dls, clip_tokenizer = get_dls(cids, valid_cids, size, bs)
    if rank_distrib() == 0: print(len(dls.train_ds), len(dls.valid_ds))
        
    # callbacks
    ndata = len(dls.train_ds)//1000
    modelname = f'XXX_shard16_{ndata}K_en_{arch}_bs{bs}_size{size}_epochs{epochs}_lr{lr}'
    savemodel_cb =  SaveModelCallback(monitor="retrieval_at_20", comp=np.greater, fname=modelname)
    if num_distrib()>0: 
        print("Distributed training mode")
        clip_trainer_cb = CLIPTrainer()
    else:
        print("Single gpu training mode")
        clip_trainer_cb = CLIPTrainer()
    cbs = [savemodel_cb, clip_trainer_cb]
    if rank_distrib() == 0 and WANDB: cbs += [WandbCallback(log_preds=False, log_model=False)]
        
    
    # ZeRO
    def zero(params, lr, **kwargs):
        return OptimWrapper(ZeroRedundancyOptimizer(params, optimizer_class=torch.optim.Adam, lr=lr))
    
    if opt == 'zero':      opt_func = zero
    elif opt == 'ranger':  opt_func = ranger
    else:                  opt_func = Adam
        
        
    # model
    if arch == 'vitb32':
        print(arch, use_grad_check, type(use_grad_check), grad_check_nchunks)
        vitb32_config_dict = vitb32_config(size, clip_tokenizer.context_length, clip_tokenizer.vocab_size)
        clip_model = CLIP(**vitb32_config_dict, checkpoint=use_grad_check, checkpoint_nchunks=grad_check_nchunks)
        if do_finetune:
            print("Loading pretrained model..")
            clip_pretrained_model, _ = clip.load(finetune_modelname, jit=False)
            clip_model.load_state_dict(clip_pretrained_model.state_dict())
    
    elif arch == 'vitl14':
        vitl14_config_dict = vitl14_config(size, clip_tokenizer.context_length, clip_tokenizer.vocab_size)
        clip_model = CLIP(**vitl14_config_dict, checkpoint=use_grad_check, checkpoint_nchunks=grad_check_nchunks)
        if do_finetune:
            raise Exception(f"No pretrained model available for arch {arch}")
    
    else: raise Exception("No matching arch.")
    
    learner = Learner(dls, clip_model, loss_func=noop, cbs=cbs, opt_func=opt_func,
                  metrics=[RetrievalAtK(k=5), 
                           RetrievalAtK(k=20), 
                           RetrievalAtK(k="mean"),
                           RetrievalAtK(k="median")])
    learner.to_fp16()

    
    # fit 
    if num_distrib()>0:
        with learner.distrib_ctx():
            print(f"num_distrib(): {num_distrib()}")
            learner.fit_flat_cos(epochs, lr, pct_start=0.25)
    else:   learner.fit_flat_cos(epochs, lr, pct_start=0.25)
    
    # end wandb
    if rank_distrib() == 0: wandb.finish()


