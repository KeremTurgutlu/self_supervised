# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10b-simclr_v2.ipynb (unless otherwise specified).

__all__ = ['create_encoder', 'create_projection_head', 'SimCLRv2Model', 'create_simclrv2_model', 'remove_diag',
           'SimCLRv2Loss', 'SimCLRv2']

# Cell
from fastai.vision.all import *
from .augmentations import *

# Cell
def create_encoder(arch, n_in=3, pretrained=True, cut=None, concat_pool=True):
    "Create encoder from a given arch backbone"
    encoder = create_body(arch, n_in, pretrained, cut)
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    return nn.Sequential(*encoder, pool, Flatten())

# Cell
def create_projection_head(dim, hidden_size=None, projection_size=256):
    "creates MLP module as described in paper"
    if not hidden_size: hidden_size=dim
    return nn.Sequential(nn.Linear(dim, hidden_size),
                         nn.ReLU(inplace=True),
                         nn.Linear(dim, hidden_size),
                         nn.ReLU(inplace=True),
                         nn.Linear(hidden_size, projection_size))

# Cell
class SimCLRv2Model(Module):
    "Compute predictions of concatenated xi and xj"
    def __init__(self,encoder,projector): self.encoder,self.projector = encoder,projector
    def forward(self,x): return self.projector(self.encoder(x))

# Cell
def create_simclrv2_model(encoder=None, arch=xresnet101, n_in=3, pretrained=True, cut=None, concat_pool=True, projection_size=256):
    "Create SimCLR from a given arch"
    if encoder is None: encoder = create_encoder(arch, n_in, pretrained, cut, concat_pool)
    with torch.no_grad(): representation = encoder(torch.randn((2,n_in,128,128)))
    projector = create_projection_head(representation.size(1), projection_size=projection_size)
    apply_init(projector)
    return SimCLRv2Model(encoder, projector)

# Cell
def remove_diag(x):
    bs = x.shape[0]
    return x[~torch.eye(bs).bool()].reshape(bs,bs-1)

# Cell
class SimCLRv2Loss(Module):
    "NT-Xent loss function"
    def __init__(self, temp=0.1):
        self.temp = temp

    def forward(self, inp, targ):
        bs,feat = inp.shape
        csim = F.cosine_similarity(inp, inp.unsqueeze(dim=1), dim=-1)/self.temp
        csim = remove_diag(csim)
        targ = remove_diag(torch.eye(targ.shape[0], device=inp.device)[targ]).nonzero()[:,-1]
        return F.cross_entropy(csim, targ)

# Cell
class SimCLRv2(Callback):
    "SimCLR callback"
    order,run_valid = 9,True
    # before to_fp16() aka MixedPrecision
    def __init__(self, size, aug_func, **aug_kwargs):
        self.aug1 = aug_func(size, **aug_kwargs)
        self.aug2 = aug_func(size, **aug_kwargs)
        print(self.aug1, self.aug2)

    def before_batch(self):
        xi,xj = self.aug1(self.x.clone()), self.aug2(self.x.clone())
        self.learn.xb = (torch.cat([xi, xj]),)
        bs = self.learn.xb[0].shape[0]
        self.learn.yb = (torch.arange(bs, device=self.dls.device).roll(bs//2),)

    def show_one(self):
        xb = TensorImage(self.learn.xb[0])
        bs = len(xb)//2
        i = np.random.choice(bs)
        xb = self.aug1.decode(xb.to('cpu').clone()).clamp(0,1)
        images = [xb[i], xb[bs+i]]
        show_images(images)