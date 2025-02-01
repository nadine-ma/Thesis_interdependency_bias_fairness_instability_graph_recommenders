import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

ltensor = torch.LongTensor


class NodeClassification(Dataset):
    def __init__(self,data_split):
        self.dataset = np.ascontiguousarray(data_split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
        data = np.ascontiguousarray(data)
        self.dataset = ltensor(data)


class PredBias(Dataset):
    def __init__(self,use_1M,movies,users,attribute,prefetch_to_gpu=False):
        self.prefetch_to_gpu = prefetch_to_gpu
        self.dataset = np.ascontiguousarray(movies)
        self.users = users
        self.groups = defaultdict(list)
        if attribute == 'gender':
            users_sex = self.users['sex']
            self.num_groups = 2
            [self.groups[val].append(ind) for ind,val in enumerate(users_sex)]
        elif attribute == 'occupation':
            users_occupation = self.users['occupation']
            if use_1M:
                [self.groups[val].append(ind) for ind,val in \
                        enumerate(users_occupation)]
                self.num_groups = 21
            else:
                users_occupation_list = sorted(set(users_occupation))
                occ_to_idx = {}
                for i, occ in enumerate(users_occupation_list):
                    occ_to_idx[occ] = i
                users_occupation = [occ_to_idx[occ] for occ in users_occupation]
        elif attribute == 'random':
            users_random = self.users['rand']
            self.num_groups = 2
            [self.groups[val].append(ind) for ind,val in enumerate(users_random)]
        else:
            users_age = self.users['age'].values
            users_age_list = sorted(set(users_age))
            if not use_1M:
                bins = np.linspace(5, 75, num=15, endpoint=True)
                inds = np.digitize(users_age, bins) - 1
                self.users_sensitive = np.ascontiguousarray(inds)
            else:
                reindex = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
                self.num_groups = 7
                inds = [reindex.get(n, n) for n in users_age]
                [self.groups[val].append(ind) for ind,val in enumerate(inds)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def shuffle(self):
        if self.dataset.is_cuda:
            self.dataset = self.dataset.cpu()

        data = self.dataset
        np.random.shuffle(data)
        data = np.ascontiguousarray(data)
        self.dataset = ltensor(data)

        if self.prefetch_to_gpu:
            self.dataset = self.dataset.cuda().contiguous()

def compute_rank(enrgs, target, mask_observed=None):
    enrg = enrgs[target]
    if mask_observed is not None:
        mask_observed[target] = 0
        enrgs = enrgs + 100*mask_observed

    return (enrgs < enrg).sum() + 1


def create_or_append(d, k, v, v2np=None):
    if v2np is None:
        if k in d:
            d[k].append(v)
        else:
            d[k] = [v]
    else:
        if k in d:
            d[k].append(v2np(v))
        else:
            d[k] = [v2np(v)]

def to_multi_gpu(model):
    cuda_stat = torch.cuda.is_available()
    if cuda_stat:
        model = torch.nn.DataParallel(model,\
                device_ids=range(torch.cuda.device_count())).cuda()
    return model
