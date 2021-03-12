# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/12 - byol.ipynb (unless otherwise specified).

__all__ = ['BYOLModel', 'create_byol_model', 'get_byol_aug_pipelines', 'BYOL']

# Cell
from fastai.vision.all import *
from ..augmentations import *
from ..layers import *

# Cell
class BYOLModel(Module):
    "Compute predictions of v1 and v2"
    def __init__(self,encoder,projector,predictor):
        self.encoder,self.projector,self.predictor = encoder,projector,predictor

    def forward(self,v1,v2):
        "Symmetric predictions for symmetric loss calc"
        q1 = self.predictor(self.projector(self.encoder(v1)))
        q2 = self.predictor(self.projector(self.encoder(v2)))
        return (q1,q2)

# Cell
def create_byol_model(encoder, hidden_size=4096, projection_size=256):
    "Create BYOL model"
    n_in  = in_channels(encoder)
    with torch.no_grad(): representation = encoder(torch.randn((2,n_in,128,128)))
    projector = create_mlp_module(representation.size(1), hidden_size, projection_size, bn=True)
    predictor = create_mlp_module(projection_size, hidden_size, projection_size, bn=True)
    apply_init(projector)
    apply_init(predictor)
    return BYOLModel(encoder, projector, predictor)

# Cell
@delegates(get_multi_aug_pipelines)
def get_byol_aug_pipelines(size, **kwargs): return get_multi_aug_pipelines(n=2, size=size, **kwargs)

# Cell
from copy import deepcopy

class BYOL(Callback):
    order,run_valid = 9,True
    def __init__(self, m=0.999, aug_pipelines=[], print_augs=False):
        assert_aug_pipelines(aug_pipelines)
        self.aug1, self.aug2 = aug_pipelines
        if print_augs: print(self.aug1), print(self.aug2)
        store_attr("m")

    def before_fit(self):
        "Create target model"
        self.target_model = deepcopy(self.learn.model).to(self.dls.device)
        for param_k in self.target_model.parameters(): param_k.requires_grad = False
        self.learn.loss_func = self.lf

    def before_batch(self):
        "Generate 2 views of the same image and calculate target projections for these views"
        v1,v2 = self.aug1(self.x), self.aug2(self.x.clone())
        self.learn.xb = (v1,v2)

        with torch.no_grad():
            z1 = self.target_model.projector(self.target_model.encoder(v1))
            z2 = self.target_model.projector(self.target_model.encoder(v2))
            self.learn.yb = (z1,z2)


    def _mse_loss(self, x, y):
        x,y = F.normalize(x), F.normalize(y)
        return 2 - 2 * (x * y).sum(dim=-1)


    def lf(self, pred, *yb):
        (q1,q2),(z1,z2) = pred,yb
        return (self._mse_loss(q1,z2) + self._mse_loss(q2,z1)).mean()


    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_k in zip(self.learn.model.parameters(), self.target_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def after_step(self):
        "Momentum update target model"
        self._momentum_update_target_encoder()


    @torch.no_grad()
    def show(self, n=1):
        x1,x2  = self.learn.xb
        bs = x1.size(0)
        idxs = np.random.choice(range(bs),n,False)
        x1 = self.aug1.decode(x1[idxs].to('cpu').clone()).clamp(0,1)
        x2 = self.aug2.decode(x2[idxs].to('cpu').clone()).clamp(0,1)
        images = []
        for i in range(n): images += [x1[i],x2[i]]
        return show_batch(x1[0], None, images, max_n=n * n, ncols=None, nrows=n)