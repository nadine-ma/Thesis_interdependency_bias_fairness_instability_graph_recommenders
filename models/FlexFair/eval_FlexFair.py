from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
import numpy as np
import sys, os
from tqdm import tqdm

from models.FlexFair.utils import PredBias

tqdm.monitor_interval = 0
sys.path.append('../')


def optimizer(params, mode, *args, **kwargs):
    if mode == 'SGD':
        opt = optim.SGD(params, *args, momentum=0., **kwargs)
    elif mode.startswith('nesterov'):
        momentum = float(mode[len('nesterov'):])
        opt = optim.SGD(params, *args, momentum=momentum, nesterov=True, **kwargs)
    elif mode.lower() == 'adam':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True,
                weight_decay=1e-4, **kwargs)
    elif mode.lower() == 'adam_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_hyp3':
        betas = kwargs.pop('betas', (0., .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_sparse':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.SparseAdam(params, *args, weight_decay=1e-4, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp3':
        betas = kwargs.pop('betas', (.0, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    else:
        raise NotImplementedError()
    return opt

def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
    y_test = np.asarray(y_test).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_pred, average=average)

def test_dummy(args,test_dataset,modelD,net,dummy,experiment,\
        epoch,strategy,multi_class=False,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=604)
    correct = 0
    preds_list, probs_list, labels_list = [], [],[]
    sensitive_attr = net.users_sensitive
    for p_batch in test_loader:
        p_batch_var = Variable(p_batch)
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y = sensitive_attr[p_batch]
        preds = dummy.predict(p_batch_emb)
        acc = 100.* accuracy_score(y,preds)
        preds_list.append(preds)
        if multi_class:
            probs_list.append(dummy.predict_proba(p_batch_emb))
        else:
            probs_list.append(dummy.predict_proba(p_batch_emb)[:, 1])
        labels_list.append(y)
    f1 = f1_score(labels_list[0],preds_list[0],average="micro")
    print("Test Dummy %s Accuracy is: %f  F1: %f" %(strategy,acc,f1))

def test_age(test_dataset,modelD,net, epoch,filter_set=None):
    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=512)
    correct = 0
    preds_list, labels_list, probs_list = [], [],[]
    for p_batch in tqdm(test_loader, position=0, leave=True, desc="Eval Age"):
        p_batch_var = Variable(p_batch)
        p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
        y_hat, y = net.predict(p_batch_emb,p_batch_var)
        preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
        correct += preds.eq(y.view_as(preds)).sum().item()
        preds_list.append(preds)
        probs_list.append(y_hat)
        labels_list.append(y)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_probs_list = torch.cat(probs_list,0).data.cpu().numpy()
    acc = 100. * correct / len(test_dataset)
    f1 = f1_score(cat_labels_list, cat_preds_list, average='micro')
    return acc, f1

def train_age(args,modelD, fairD_age,opt,train_dataset,test_dataset,attr_data,\
        experiment,filter_set=None):
    modelD.eval()
    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=3000)
    criterion = nn.NLLLoss()

    for epoch in range(1,args.num_classifier_epochs+1):
        correct = 0
        if epoch % 1 == 0:
            acc, f1 = test_age(args,test_dataset,modelD,fairD_age,experiment,epoch,filter_set)
            print(f"Test Age Accuracy Epoch {epoch} is: {acc} F1: {f1}")
        embs_list, labels_list = [], []
        for p_batch in train_loader:
            p_batch_var = Variable(p_batch)
            p_batch_emb = modelD.encode(p_batch_var.detach(),filter_set)
            opt.zero_grad()
            y_hat, y = fairD_age(p_batch_emb,p_batch_var)
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            preds = y_hat.max(1, keepdim=True)[1] # get the index of the max
            correct = preds.eq(y.view_as(preds)).sum().item()
            acc = 100. * correct / len(p_batch)
            #AUC = multiclass_roc_auc_score(y.data.cpu().numpy(),y_hat.data.cpu().numpy())
            f1 = f1_score(y.data.cpu().numpy(), preds.data.cpu().numpy(), average='micro')
            if epoch == args.num_classifier_epochs:
                embs_list.append(p_batch_emb)
                labels_list.append(y)
            print("Train Age Loss is %f Accuracy is: %f  F1: %f" \
                    %(loss,acc,f1))

    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    cat_embs_list = torch.cat(embs_list,0).data.cpu().numpy()
    ''' Dummy Classifier '''
    for strategy in ['stratified', 'most_frequent', 'uniform']:
        dummy = DummyClassifier(strategy=strategy)
        dummy.fit(cat_embs_list, cat_labels_list)
        test_dummy(args,test_dataset,modelD,fairD_age,dummy,experiment,\
                epoch,strategy,True,filter_set)


def onevsall_bias(vals,pos_index):
    bias = 0
    for i in range(0,len(vals)):
        bias = torch.abs(vals[pos_index] - vals[i])
    weighted_avg_bias = bias / len(vals)
    return weighted_avg_bias
def calc_majority_class(groups,attribute):
    counts = []
    for k in groups.keys():
        counts.append(len(groups[k]))
    counts = np.asarray(counts)
    index = np.argmax(counts)
    prob = 100. * np.max(counts) / counts.sum()
    print("%s Majority Class %s has prob %f" %(attribute,index,prob))

def calc_attribute_bias(mode,args,modelD,experiment,\
        attribute,epoch,filter_=None):
    movie_ids = args.movies['movie_id'].values
    if mode == 'Train':
        dataset = PredBias(args.use_1M,movie_ids,args.users[:args.cutoff_row],\
                attribute, args.prefetch_to_gpu)
    else:
        dataset = PredBias(args.use_1M,movie_ids,args.users[args.cutoff_row:],\
                attribute,args.prefetch_to_gpu)

    data_loader = DataLoader(dataset, num_workers=1, batch_size=4000)
    if args.show_tqdm:
        test_data_itr = tqdm(enumerate(data_loader))
    else:
        test_data_itr = enumerate(data_loader)

    groups = dataset.groups
    group_preds = defaultdict(list)
    group_embeds_list = []
    calc_majority_class(groups,attribute)
    for idx, movies in test_data_itr:
        movies_var = Variable(movies)
        with torch.no_grad():
            movies_embed = modelD.encode(movies_var)
            for group, vals in groups.items():
                users_var = Variable(torch.LongTensor(vals))
                users_embed = modelD.encode(users_var,filter_)
                group_embeds_list.append(users_embed)

    for group_embed in group_embeds_list:
        movies_repeated = movies_embed.repeat(len(group_embed),1,1).permute(1,0,2)
        with torch.no_grad():
            for i in range(0,len(movies_repeated)):
                preds = modelD.decoder.predict(group_embed,movies_repeated[i])
                avg_preds = preds.mean()
                group_preds[i].append(avg_preds)

    bias = 0
    for ind, val in group_preds.items():
        if len(val) == 2:
            bias += torch.abs(val[0] - val[1])
        else:
            weighted_bias = 0
            for i in range(0,len(val)):
                weighted_bias += onevsall_bias(val,i)
            bias += weighted_bias / len(val)
    avg_bias = bias / len(movies)
    print("%s %s Bias is %f" %(mode,attribute,avg_bias))
    if args.do_log:
        experiment.log_metric(mode +" " + attribute + "Bias",float(avg_bias))
    return avg_bias


def train_fairness_classifier_gcmc(train_dataset,args,modelD,experiment,fairD,\
        fair_optim,epoch,filter_=None,retrain=False,log_freq=2):

    train_loader = DataLoader(train_dataset, num_workers=1, batch_size=8000)#, collate_fn=collate_fn)
    correct = 0
    total_ent = 0
    gender_correct,occupation_correct,age_correct,random_correct = 0,0,0,0
    preds_list = []
    labels_list = []

    # train_data_itr = tqdm(enumerate(train_loader))
    train_data_itr = enumerate(train_loader)

    ''' Training Classifier on Nodes '''
    # for epoch in tqdm(range(1, args.num_classifier_epochs + 1)):
    for idx, p_batch in train_data_itr:
        fair_optim.zero_grad()
        p_batch_var = Variable(p_batch)
        p_batch_emb = modelD.encode(p_batch_var)
        if filter_ is not None:
            p_batch_emb = filter_(p_batch_emb)
        fairD_loss = fairD(p_batch_emb.detach(),p_batch_var)
        print("%s Classifier has loss %f" %(fairD.attribute,fairD_loss))
        fairD_loss.backward(retain_graph=False)
        fair_optim.step()
        with torch.no_grad():
            l_preds, l_A_labels, probs = fairD.predict(p_batch_emb,\
                    p_batch_var,return_preds=True)
            l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
            if fairD.attribute == 'gender':
                fairD_gender_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='binary')
                gender_correct += l_correct #
            elif fairD.attribute == 'occupation':
                fairD_occupation_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='micro')
                occupation_correct += l_correct
            elif fairD.attribute == 'age':
                fairD_age_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='micro')
                age_correct += l_correct
            else:
                fairD_random_loss = fairD_loss.detach().cpu().numpy()
                l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                        average='micro')
                random_correct += l_correct

            preds_list.append(probs)
            labels_list.append(l_A_labels.view_as(l_preds))
            correct += l_correct
            total_ent += len(p_batch)

        acc = 100. * correct / total_ent
        print('Train Accuracy is %f' %(float(acc)))
        cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
        cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
        print('Train Accuracy is %f' %(float(acc)))
        if fairD.attribute == 'gender' or fairD.attribute == 'random':
            AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
            print('Train AUC is %f' %(float(AUC)))
        else:
            AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
            print('Train AUC is %f' %(float(AUC)))

        ''' Logging '''
        if args.do_log:
            counter = epoch
            acc = 100. * correct / total_ent
            experiment.log_metric(fairD.attribute + " Train FairD Accuracy",float(acc),step=counter)
            if fairD.attribute == 'gender' or fairD.attribute == 'random':
                cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
                cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
                print('Train Accuracy is %f' %(float(acc)))
                if fairD.attribute == 'gender' or fairD.attribute == 'random':
                    AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
                    # AUC = roc_auc_score(cat_labels_list, cat_preds_list)
                    print('Train AUC is %f' %(float(AUC)))
                    experiment.log_metric(fairD.attribute + " Train FairD AUC",float(AUC),step=counter)
                # else:
                    # AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
                    # print('Train AUC is %f' %(float(AUC)))
                    # experiment.log_metric(fairD.attribute + " Train FairD AUC",float(AUC),step=counter)
            if fairD.attribute == 'gender':
                experiment.log_metric("Train Classifier Gender Disc Loss",float(fairD_gender_loss),step=counter)
            if fairD.attribute == 'occupation':
                experiment.log_metric("Train Classifier Occupation Disc Loss",float(fairD_occupation_loss),step=counter)
            if fairD.attribute == 'age':
                experiment.log_metric("Train Classifier Age Disc Loss",float(fairD_age_loss),step=counter)
            if fairD.attribute == 'random':
                experiment.log_metric("Train Classifier  Random Disc Loss",float(fairD_random_loss),step=counter)

def train_fairness_classifier(dataset,args,modelD,experiment,fairD,\
        fair_optim,epoch,filter_=None,retrain=False):
    train_fairness_classifier_gcmc(dataset,args,modelD,experiment,fairD,\
        fair_optim,epoch,filter_=None,retrain=False,log_freq=2)


def test_fairness_gcmc(test_dataset,args,modelD,experiment,fairD,\
        attribute,epoch,filter_=None,retrain=False):

    test_loader = DataLoader(test_dataset, num_workers=1, batch_size=8000)#, collate_fn=collate_fn)
    correct = 0
    total_ent = 0
    precision_list = []
    recall_list = []
    fscore_list = []
    preds_list = []
    labels_list = []

    if args.show_tqdm:
        test_data_itr = tqdm(enumerate(test_loader))
    else:
        test_data_itr = enumerate(test_loader)

    ''' Test Classifier on Nodes '''
    for idx, p_batch in test_data_itr:
        p_batch_var = Variable(p_batch)
        p_batch_emb = modelD.encode(p_batch_var)
        ''' If Compositional Add the Attribute Specific Filter '''
        if filter_ is not None:
            p_batch_emb = filter_(p_batch_emb)
        l_preds, l_A_labels, probs = fairD.predict(p_batch_emb,\
                p_batch_var,return_preds=True)
        l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
        correct += l_correct
        if fairD.attribute == 'gender':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='binary')
        elif fairD.attribute == 'occupation':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')
        elif fairD.attribute == 'age':
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')
        else:
            l_precision,l_recall,l_fscore,_ = precision_recall_fscore_support(l_A_labels, l_preds,\
                    average='micro')

        precision = l_precision
        recall = l_recall
        fscore = l_fscore
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
        preds_list.append(probs)
        labels_list.append(l_A_labels.view_as(l_preds))

    acc = 100. * correct / len(test_dataset)
    cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
    cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
    print('Classifier Test Accuracy is %f' %(float(acc)))
    if fairD.attribute == 'gender' or fairD.attribute == 'random':
        AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
        print('Classifier Test AUC is %f' %(float(AUC)))
    else:
        AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
        print('Classifier Test AUC is %f' %(float(AUC)))

    ''' Logging '''
    if args.do_log:
        acc = 100. * correct / len(test_dataset)
        experiment.log_metric(fairD.attribute + " Test FairD Accuracy",float(acc),step=epoch)
        cat_preds_list = torch.cat(preds_list,0).data.cpu().numpy()
        cat_labels_list = torch.cat(labels_list,0).data.cpu().numpy()
        print('Classifier Test Accuracy is %f' %(float(acc)))
        if fairD.attribute == 'gender' or fairD.attribute == 'random':
            AUC = roc_auc_score(cat_labels_list, np.argmax(cat_preds_list,1))
            # AUC = roc_auc_score(cat_labels_list, cat_preds_list)
            print('Classifier Test AUC is %f' %(float(AUC)))
            experiment.log_metric(fairD.attribute + " Test FairD AUC",float(AUC),step=epoch)
        else:
            AUC = multiclass_roc_auc_score(cat_labels_list,cat_preds_list)
            print('Classifier Test AUC is %f' %(float(AUC)))
            experiment.log_metric(fairD.attribute + " Test FairD AUC",float(AUC),step=epoch)

