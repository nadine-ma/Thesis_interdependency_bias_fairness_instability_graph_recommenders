import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.autograd import Variable
import numpy as np
import sys, os

from tqdm import tqdm
tqdm.monitor_interval = 0


sys.path.append('../')


''' Some Helpful Globals '''
ftensor = torch.FloatTensor
ltensor = torch.LongTensor
v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = False

def apply_filters_gcmc(p_lhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb, filter_r_emb = 0,0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_l_emb += filter_(p_lhs_emb)
    return filter_l_emb

def apply_filters_single_node(p_lhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb, filter_r_emb = 0,0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_l_emb += filter_(p_lhs_emb)
    return filter_l_emb


def apply_filters_transd(p_lhs_emb,p_rhs_emb,masked_filter_set):
    ''' Doesnt Have Masked Filters yet '''
    filter_l_emb = 0
    filter_r_emb = 0
    for filter_ in masked_filter_set:
        if filter_ is not None:
            filter_l_emb += filter_(p_lhs_emb)
            filter_r_emb += filter_(p_rhs_emb)
    return filter_l_emb,filter_r_emb

class TransD(nn.Module):
    def __init__(self, num_ent, num_rel, embed_dim, p):
        super(TransD, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.embed_dim = embed_dim
        self.p = p

        r = 6 / np.sqrt(self.embed_dim)

        self._ent_embeds = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self.ent_transfer = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_transfer = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self._ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
        self.rel_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)

    def transfer(self, emb, e_transfer, r_transfer):
        return emb + (emb * e_transfer).sum(dim=1, keepdim=True) * r_transfer

    #@profile
    def ent_embeds(self, idx, rel_idx):
        es = self._ent_embeds(idx)
        ts = self.ent_transfer(idx)

        rel_ts = self.rel_transfer(rel_idx)
        proj_es = self.transfer(es, ts, rel_ts)
        return proj_es

    def forward(self, triplets, return_ent_emb=False, filters=None):
        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]

        rel_es = self.rel_embeds(rel_idxs)

        lhs = self.ent_embeds(lhs_idxs, rel_idxs)
        rhs = self.ent_embeds(rhs_idxs, rel_idxs)
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                lhs,rhs = apply_filters_transd(lhs,rhs,filters)

        if not return_ent_emb:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs
        else:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs,lhs,rhs

    def get_embed(self, ents, rel_idxs, filters=None):
        with torch.no_grad():
            ent_embed = self.ent_embeds(ents, rel_idxs)
            if filters is not None:
                constant = len(filters) - filters.count(None)
                if constant !=0:
                    ent_embed = apply_filters_single_node(ent_embed,filters)
        return ent_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class TransD_BiDecoder(nn.Module):
    def __init__(self, num_ent, num_rel, embed_dim, p):
        super(TransD_BiDecoder, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.embed_dim = embed_dim
        self.p = p

        r = 6 / np.sqrt(self.embed_dim)

        ''' Encoder '''
        self._ent_embeds = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_embeds = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self.ent_transfer = nn.Embedding(self.num_ent, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)
        self.rel_transfer = nn.Embedding(self.num_rel, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

        self._ent_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)
        self.rel_embeds.weight.data.uniform_(-r, r)#.renorm_(p=2, dim=1, maxnorm=1)

        ''' Decoder '''
        self.decoder = nn.Embedding(self.embed_dim, self.embed_dim, max_norm=1, norm_type=2, sparse=USE_SPARSE_EMB)

    def transfer(self, emb, e_transfer, r_transfer):
        return emb + (emb * e_transfer).sum(dim=1, keepdim=True) * r_transfer

    #@profile
    def ent_embeds(self, idx, rel_idx):
        es = self._ent_embeds(idx)
        ts = self.ent_transfer(idx)

        rel_ts = self.rel_transfer(rel_idx)
        proj_es = self.transfer(es, ts, rel_ts)
        return proj_es

    def forward(self, triplets, return_ent_emb=False):
        lhs_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        rhs_idxs = triplets[:, 2]

        rel_es = self.rel_embeds(rel_idxs)

        lhs = self.ent_embeds(lhs_idxs, rel_idxs)
        rhs = self.ent_embeds(rhs_idxs, rel_idxs)

        if not return_ent_emb:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs
        else:
            enrgs = (lhs + rel_es - rhs).norm(p=self.p, dim=1)
            return enrgs,lhs,rhs

    def get_embed(self, ents, rel_idxs):
        ent_embed = self.ent_embeds(ents, rel_idxs)
        return ent_embed

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))


class AttributeFilter(nn.Module):
    def __init__(self, embed_dim, attribute='gender'):
        super(AttributeFilter, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute = attribute
        self.W1 = nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True)
        self.W2 = nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True)
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)

    def forward(self, ents_emb):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h2 = self.batchnorm(h2)
        return h2

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))


class BilinearDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, num_relations, embed_dim):
        super(BilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_relations, embed_dim*embed_dim)
        self.embed_dim = embed_dim

    def forward(self, embeds1, embeds2, rels):
        rel_mats = self.rel_embeds(rels).reshape(-1, self.embed_dim, self.embed_dim)
        embeds1 = torch.matmul(embeds1, rel_mats)
        return (embeds1 * embeds2).sum(dim=1)


class SharedBilinearDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """

    def __init__(self, num_relations, num_weights, embed_dim):
        super(SharedBilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_weights, embed_dim*embed_dim)
        self.weight_scalars = nn.Parameter(torch.Tensor(num_weights,num_relations))
        stdv = 1. / math.sqrt(self.weight_scalars.size(1))
        self.weight_scalars.data.uniform_(-stdv, stdv)
        self.embed_dim = embed_dim
        self.num_weights = num_weights
        self.num_relations = num_relations
        self.nll = nn.NLLLoss()
        self.mse = nn.MSELoss()

    def predict(self,embeds1,embeds2):
        basis_outputs = []
        for i in range(0,self.num_weights):
            index = Variable(torch.LongTensor([i]))
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim,\
                    self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q*embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs,dim=1)
        logit = torch.matmul(basis_outputs,self.weight_scalars)
        outputs = F.log_softmax(logit,dim=1)
        preds = 0
        for j in range(0,self.num_relations):
            index = Variable(torch.LongTensor([j]))
            ''' j+1 because of zero index '''
            preds += (j+1)*torch.exp(torch.index_select(outputs, 1,index))
        return preds

    def forward(self, embeds1, embeds2, rels):
        basis_outputs = []
        for i in range(0,self.num_weights):
            index = Variable(torch.LongTensor([i]))
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim,\
                    self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q*embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs,dim=1)
        logit = torch.matmul(basis_outputs,self.weight_scalars)
        outputs = F.log_softmax(logit,dim=1)
        log_probs = torch.gather(outputs,1,rels.unsqueeze(1))
        loss = self.nll(outputs,rels)
        preds = 0
        for j in range(0,self.num_relations):
            index = Variable(torch.LongTensor([j]))
            ''' j+1 because of zero index '''
            preds += (j+1)*torch.exp(torch.index_select(outputs, 1,index))
        return loss,preds

class SimpleGCMC(nn.Module):
    def __init__(self, decoder, embed_dim, num_ent, encoder=None, attr_filter=None):
        super(SimpleGCMC, self).__init__()
        self.attr_filter = attr_filter
        self.decoder = decoder
        self.num_ent = num_ent
        self.embed_dim = embed_dim
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        if encoder is None:
            r = 6 / np.sqrt(self.embed_dim)
            self.encoder = nn.Embedding(self.num_ent, self.embed_dim,\
                    max_norm=1, norm_type=2)
            self.encoder.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)
        else:
            self.encoder = encoder

    def predict_uim(self, user_id, num_users, num_items, return_embeds=False, filters=None):
        pos_tail_embeds = self.encode(torch.tensor([i + num_users for i in range(num_items)]))#, batchnorm=False)

        user_pos_head_embeds = []
        rels = []
        for item in pos_tail_embeds:
            user_pos_head_embeds.append(user_id)
            rels.append(0)
        user_pos_head_embeds = self.encode(torch.tensor(user_pos_head_embeds))#, batchnorm=False)
        rels = torch.tensor(rels)
        _, preds = self.decoder(user_pos_head_embeds, pos_tail_embeds, rels)

        return preds

    def encode(self, nodes, filters=None):
        embs = self.encoder(nodes.long())
        embs = self.batchnorm(embs)
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                embs = apply_filters_gcmc(embs,filters)
        return embs


    def encode_no_norm(self, nodes, filters=None):
        embs = self.encoder(nodes.long())
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                embs = apply_filters_gcmc(embs,filters)
        return embs

    def predict_rel(self,heads,tails,filters=None):
        with torch.no_grad():
            head_embeds = self.encode(heads)
            tails_embed = self.encode(tails)
            if filters is not None:
                constant = len(filters) - filters.count(None)
                if constant !=0:
                    head_embeds = apply_filters_gcmc(head_embeds,filters)
            preds = self.decoder.predict(head_embeds,tails_embed)
        return preds

    def forward_no_norm(self, pos_edges, weights=None, return_embeds=False, filters=None):
        pos_head_embeds = self.encode_no_norm(pos_edges[:,0])
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                pos_head_embeds = apply_filters_gcmc(pos_head_embeds,filters)
        pos_tail_embeds = self.encode_no_norm(pos_edges[:,-1])
        rels = pos_edges[:,1]
        loss, preds = self.decoder(pos_head_embeds, pos_tail_embeds, rels)
        if return_embeds:
            return loss, preds, pos_head_embeds, pos_tail_embeds
        else:
            return loss, preds


    def forward(self, pos_edges, weights=None, return_embeds=False, filters=None):
        pos_head_embeds = self.encode(pos_edges[:,0])
        if filters is not None:
            constant = len(filters) - filters.count(None)
            if constant !=0:
                pos_head_embeds = apply_filters_gcmc(pos_head_embeds,filters)
        pos_tail_embeds = self.encode(pos_edges[:,-1])
        rels = pos_edges[:,1]
        loss, preds = self.decoder(pos_head_embeds, pos_tail_embeds, rels)
        if return_embeds:
            return loss, preds, pos_head_embeds, pos_tail_embeds
        else:
            return loss, preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))


class AgeDiscriminator(nn.Module):
    def __init__(self,embed_dim,attribute_data,attribute,use_cross_entropy=True):
        super(AgeDiscriminator, self).__init__()
        self.embed_dim = int(embed_dim)
        self.attribute_data = attribute_data
        self.attribute = attribute
        self.criterion = nn.NLLLoss()
        if use_cross_entropy:
            self.cross_entropy = True
        else:
            self.cross_entropy = False

        users_age = attribute_data
        users_age_list = sorted(set(users_age))

        reindex = {1:0, 18:1, 25:2, 35:3, 45:4, 50:5, 56:6}
        inds = [reindex.get(n, n) for n in users_age]
        self.users_sensitive = np.ascontiguousarray(inds)
        self.out_dim = 7

        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2 ), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self, ents_emb, ents, return_loss=False):
        scores = self.net(ents_emb)
        output = F.log_softmax(scores, dim=1)
        A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()]))
        if return_loss:
            loss = self.criterion(output.squeeze(), A_labels)
            return loss
        else:
            return output.squeeze(),A_labels

    def predict(self, ents_emb, ents, return_preds=False):
        with torch.no_grad():
            scores = self.net(ents_emb)
            output = F.log_softmax(scores, dim=1)
            A_labels = Variable(torch.LongTensor(self.users_sensitive[ents.cpu()]))
            preds = output.max(1, keepdim=True)[1]
        if return_preds:
            return output.squeeze(),A_labels,preds
        else:
            return output.squeeze(),A_labels

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

