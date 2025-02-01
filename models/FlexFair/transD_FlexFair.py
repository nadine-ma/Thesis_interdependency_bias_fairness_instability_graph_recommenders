import torch
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import sys, os
from tqdm import tqdm
tqdm.monitor_interval = 0

sys.path.append('../')


ftensor = torch.FloatTensor
ltensor = torch.LongTensor
v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True


class KBDataset(Dataset):
    def __init__(self,data,user_num):
        self.dataset = data
        self.user_num = user_num
        for el in self.dataset:
            #reindexing of rating indexes
            el[2] = el[2]-1
            el[1] = el[1] + self.user_num

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        el = self.dataset[idx]
        return np.array([el[0], el[2], el[1]])


def optimizer(params, mode, *args, **kwargs):
    if mode == 'SGD':
        opt = optim.SGD(params, *args, momentum=0., **kwargs)
    elif mode.startswith('nesterov'):
        momentum = float(mode[len('nesterov'):])
        opt = optim.SGD(params, *args, momentum=momentum, nesterov=True, **kwargs)
    elif mode.lower() == 'adam':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.Adam(params, *args, betas=betas, **kwargs)
    elif mode.lower() == 'adam_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.Adam(params, *args, betas=betas, **kwargs)
    else:
        raise NotImplementedError()
    return opt


def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False


def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(np.array(batch)).contiguous()
    else:
        return torch.stack(batch).contiguous()


def mask_fairDiscriminators(discriminators, mask):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in zip(discriminators, mask) if s)

