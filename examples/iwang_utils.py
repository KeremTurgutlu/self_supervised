from fastai.vision.all import *

def get_dls(size, bs, workers=None, with_labels=False):
    path = URLs.IMAGEWANG_160 if size <= 160 else URLs.IMAGEWANG
    source = untar_data(path)
    
    def const_label(o): return 0
    
    if with_labels: y_tfms = [parent_label, Categorize()]
    else:           y_tfms = [const_label]
    
    files = get_image_files(source)
    tfms = [[PILImage.create, ToTensor, RandomResizedCrop(size, min_scale=1.)], y_tfms]
    
    dsets = Datasets(files, tfms=tfms, splits=RandomSplitter(valid_pct=0.1)(files))
    
    batch_tfms = [IntToFloatTensor]
    dls = dsets.dataloaders(bs=bs, num_workers=workers, after_batch=batch_tfms)
    return dls


def get_finetune_dls(size, bs, workers=None):
    path = URLs.IMAGEWANG_160 if size <= 160 else URLs.IMAGEWANG
    source = untar_data(path)
    files = get_image_files(source, folders=['train', 'val'])
    splits = GrandparentSplitter(valid_name='val')(files)
    
    item_aug = [RandomResizedCrop(size, min_scale=0.35), FlipItem(0.5)]
    tfms = [[PILImage.create, ToTensor, *item_aug], 
            [parent_label, Categorize()]]
    
    dsets = Datasets(files, tfms=tfms, splits=splits)
    
    batch_tfms = [IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]
    dls = dsets.dataloaders(bs=bs, num_workers=workers, after_batch=batch_tfms)
    return dls

def create_finetune_learner(size, bs, encoder_path, arch='xresnet34', checkpoint=True, n_checkpoint=2):
    # create dataloader
    dls = get_finetune_dls(size, bs=bs//2)
    # load pretrained weights
    pretrained_encoder = torch.load(encoder_path)
    encoder = create_encoder(arch, pretrained=False, n_in=3)
    if checkpoint: encoder = CheckpointSequential(encoder, checkpoint_nchunks=n_checkpoint)
    encoder.load_state_dict(pretrained_encoder)
    # create new classifier head
    nf = encoder(torch.randn(2,3,224,224)).size(-1)
    classifier = create_cls_module(nf, dls.c)
    # create model and learner
    model = nn.Sequential(encoder, classifier)
    learner = Learner(dls, model, opt_func=opt_func,
                    metrics=[accuracy,top_k_accuracy], loss_func=LabelSmoothingCrossEntropy())
    return learner