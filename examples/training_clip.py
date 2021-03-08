# ### Dataset
title_df = pd.read_csv("XXX.csv")
cid2title = dict(zip(title_df['cid'], title_df['title']))
cids = title_df['cid'].values
datapath = Path("XXX")
image_files = get_image_files(datapath)
cid2file = {int(o.stem.split("_")[2]):o for o in image_files}



def read_image(cid): return PILImage.create(cid2file[cid])
def read_text(cid): return cid2title[cid]


def dummy_targ(o): return 0 # loss func is not called without this



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




sample_cids = pd.read_pickle("XXX.pkl")
sample_valid_cids = pd.read_pickle("XXX.pkl")


# In[52]:


len(sample_cids), len(sample_valid_cids)


# ### Train

import wandb
from fastai.callback.wandb import WandbCallback
from custom_clip.model import CLIP as CustomCLIP
from fastai.distributed import *
torch.backends.cudnn.benchmark = True

@call_parse
def main(
    size:      Param("Image resolution", int)=224,    
    bs:        Param("Batch Size", int)=128,
    epochs:    Param("Number of epochs for training", int)=1,    
    lr:        Param("Learning rate for training", float)=5e-5):
    
    WANDB = True
        
    # start wandb
    if rank_distrib() == 0 and WANDB:
        wandb.init(project="XXX", entity="XXX");
        wandb.config.update({"Arch":"ViT-B/32", 
                             "Size":size,
                             "BS":bs,
                             "Compute":"Single GPU Non Distributed Loss",
                             "Training":"From Scratch"});

    # dataloaders
    dls, clip_tokenizer = get_dls(cids, sample_valid_cids[:10000], size, bs)
    if rank_distrib() == 0: print(len(dls.train_ds), len(dls.valid_ds))
        
    # callbacks
    ndata = len(dls.train_ds)//1000
    modelname = f'XX{ndata}K_en_vitb32_bs{bs}_size{size}_epochs{epochs}_lr{lr}'
    savemodel_cb =  SaveModelCallback(monitor="retrieval_at_20", comp=np.greater, fname=modelname)
    if num_distrib()>0: 
        print("Distributed training mode")
        clip_trainer_cb = DistributedCLIPTrainer()
    else:
        print("Single gpu training mode")
        clip_trainer_cb = CLIPTrainer()
    cbs = [savemodel_cb, clip_trainer_cb]
    if rank_distrib() == 0 and WANDB: cbs += [WandbCallback(log_preds=False, log_model=False)]
        

    # model
    vitb32_config_dict = vitb32_config(size, clip_tokenizer.context_length, clip_tokenizer.vocab_size)
    clip_model = CustomCLIP(**vitb32_config_dict)
    learner = Learner(dls, clip_model, loss_func=noop, cbs=cbs,
                  metrics=[RetrievalAtK(k=5), 
                           RetrievalAtK(k=20), 
                           RetrievalAtK(k="mean"),
                           RetrievalAtK(k="median")])
    learner.to_fp16()
    learner.unfreeze()
    
    # fit 
    if num_distrib()>0:
        with learner.distrib_ctx():
            print(f"num_distrib(): {num_distrib()}")
            lr *= math.sqrt(num_distrib())
            learner.fit_flat_cos(epochs, lr, pct_start=0.25)
    else:   learner.fit_flat_cos(epochs, lr, pct_start=0.25)
    
    # end wandb
    if rank_distrib() == 0: wandb.finish()


