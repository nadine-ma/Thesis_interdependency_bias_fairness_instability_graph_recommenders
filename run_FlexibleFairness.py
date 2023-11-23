import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import argparse
import sys, os
from tqdm import tqdm
import datetime

from data_utils.utils import load_random_items, generate_user_list, generate_rating_list, write_eval_measures
from data_utils.utils_data_preparation import load_user_features, id_remapping_users, \
    id_remapping_items, split_rating_data, remove_implicit_ratings_BookCrossing
from models.FlexFair.model import SharedBilinearDecoder, SimpleGCMC, AgeDiscriminator
from models.FlexFair.transD_FlexFair import collate_fn, KBDataset, optimizer
from models.FlexFair.utils import NodeClassification
from models.FlexFair.eval_FlexFair import test_age

from models.metrics import get_popularity_distribution
from models.utils import create_complete_user_item_matrix, create_uim, LF_model

tqdm.monitor_interval = 0

sys.path.append('../')
import gc

ltensor = torch.LongTensor

v2np = lambda v: v.data.cpu().numpy()
USE_SPARSE_EMB = True


def run_FlexibleFairness(hparams, instance_bool=False, gamma_bool=False):

    data_dir = hparams.data_dir
    dataset_name = hparams.dataset
    num_random_items = hparams.num_random_items
    batch_size = hparams.batch_size
    top_k = hparams.top_k
    eval_ratio = hparams.eval_ratio
    test_ratio = hparams.test_ratio
    lr = hparams.lr
    d_steps = hparams.D_steps
    embed_dim = hparams.embed_dim
    gamma = hparams.gamma
    num_epochs = hparams.num_epochs
    use_age_attr = hparams.use_age_attr
    use_cross_entropy = hparams.use_cross_entropy
    training_seed = hparams.seed


    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed)
    random.seed(training_seed)


    logdir = os.path.join("logs", dataset_name, "FlexibleFairness",
                               datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir)

    with open(os.path.join(logdir, "hparams.txt"), "w") as hparams_log_file:
        for el in hparams._get_kwargs():
            hparams_log_file.write(f'{el[0]}:{el[1]}:{type(el[1]).__name__}\n')

    if instance_bool is not False:
        with open(os.path.join("logs", dataset_name, "FlexibleFairness", f'instances_{instance_bool}.txt'), 'a') as file:
            file.write(f'{logdir}\n')

    if gamma_bool is not False:
        with open(os.path.join("logs", dataset_name, "FlexibleFairness", f'gammas_{gamma_bool}.txt'), 'a') as file:
            file.write(f'{logdir},{gamma}\n')


    use_cuda = False
    device = torch.device("cpu")

    uim_full = create_complete_user_item_matrix(data_dir, dataset_name)
    num_users = uim_full.shape[0]
    num_movies = uim_full.shape[1]
    uim_train = create_uim(data_dir, dataset_name, eval_ratio, test_ratio, train=True, eval=False)
    uim_eval = create_uim(data_dir, dataset_name, eval_ratio, test_ratio, train=False, eval=True)


    users_features = load_user_features(data_dir, dataset_name)
    train_ratings = generate_rating_list(uim_train)

    train_ratings = np.array(train_ratings)
    train_ratings = train_ratings.astype("int64")
    test_ratings = np.array(generate_rating_list(uim_eval))
    test_ratings = test_ratings.astype("int64")

    num_ent = num_users + num_movies

    num_rel = 5
    if dataset_name == "bookCrossing":
        num_rel = 10

    users_train = generate_user_list(data_dir, dataset_name, train=True, eval=False)
    users_test = generate_user_list(data_dir, dataset_name, train=False, eval=True)


    # ToDO#############################################################

    train_set = KBDataset(train_ratings, num_users)
    test_set = KBDataset(test_ratings, num_users)
    train_fairness_set = NodeClassification(users_train)
    test_fairness_set = NodeClassification(users_test)

    decoder = SharedBilinearDecoder(num_rel, 2, embed_dim).to(device)
    modelD = SimpleGCMC(decoder, embed_dim, num_ent).to(device)

    fairD_gender, fairD_occupation, fairD_age, fairD_random = None, None, None, None
    optimizer_fairD_gender, optimizer_fairD_occupation, \
        optimizer_fairD_age, optimizer_fairD_random = None, None, None, None
    gender_filter, occupation_filter, age_filter = None, None, None
    if use_age_attr:
        attr_data = [i for i in users_features]
        fairD_age = AgeDiscriminator(embed_dim, attr_data, attribute='age', use_cross_entropy=use_cross_entropy)
        optimizer_fairD_age = optimizer(fairD_age.parameters(), 'adam', lr)


    ''' Create Sets '''
    fairD_set = [fairD_gender, fairD_occupation, fairD_age, fairD_random]
    filter_set = [gender_filter, occupation_filter, age_filter, None]
    optimizer_fairD_set = [optimizer_fairD_gender, optimizer_fairD_occupation, \
                           optimizer_fairD_age, optimizer_fairD_random]

    ''' Initialize CUDA if Available '''
    if use_cuda:
        for fairD, filter_ in zip(fairD_set, filter_set):
            if fairD is not None:
                fairD.to(device)
            if filter_ is not None:
                filter_.to(device)


    optimizerD = optimizer(modelD.parameters(), 'adam', lr)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, pin_memory=True, collate_fn=collate_fn)

    ##########################################  ##########################################

    best_test_loss = 100000000
    epoch = 0

    best_RMSE = [100, 0]
    best_F1 = [100, 0]

    while epoch <= num_epochs:
        print()
        if epoch > 0:
            ################################ TRAINING #####################################
            modelD.train()
            fairD_gender_loss, fairD_occupation_loss, fairD_age_loss, \
                fairD_random_loss = 0, 0, 0, 0

            for p_batch in tqdm(train_loader, position=0, leave=True, desc=f'EPOCH {epoch} TRAINING'):
                masked_fairD_set = fairD_set
                masked_optimizer_fairD_set = optimizer_fairD_set
                masked_filter_set = filter_set
                if use_cuda:
                    p_batch = p_batch.cuda()
                p_batch_var = Variable(p_batch)

                ''' Number of Active Discriminators '''
                constant = len(masked_fairD_set) - masked_fairD_set.count(None)

                ''' Update GCMC Model '''
                if constant != 0:
                    task_loss, preds, lhs_emb, rhs_emb = modelD(p_batch_var, return_embeds=True, filters=masked_filter_set)
                    filter_l_emb = lhs_emb[:len(p_batch_var)]
                    l_penalty = 0

                    ''' Apply Discriminators '''
                    for fairD_disc, fair_optim in zip(masked_fairD_set, masked_optimizer_fairD_set):
                        if fairD_disc is not None and fair_optim is not None:
                            l_penalty += fairD_disc(filter_l_emb, p_batch[:, 0], True)

                    if not use_cross_entropy:
                        fair_penalty = constant - l_penalty
                    else:
                        fair_penalty = -1 * l_penalty

                    optimizerD.zero_grad()
                    full_loss = task_loss + gamma * fair_penalty
                    full_loss.backward(retain_graph=False)
                    optimizerD.step()

                    for k in range(0, d_steps):
                        l_penalty_2 = 0
                        for fairD_disc, fair_optim in zip(masked_fairD_set, masked_optimizer_fairD_set):
                            if fairD_disc is not None and fair_optim is not None:
                                fair_optim.zero_grad()
                                l_penalty_2 += fairD_disc(filter_l_emb.detach(), p_batch[:, 0], True)
                                if not use_cross_entropy:
                                    fairD_loss = -1 * (1 - l_penalty_2)
                                else:
                                    fairD_loss = l_penalty_2
                                fairD_loss.backward(retain_graph=True)
                                fair_optim.step()
                else:
                    task_loss, preds = modelD(p_batch_var)
                    fair_penalty = Variable(torch.zeros(1))
                    optimizerD.zero_grad()
                    full_loss = task_loss + gamma * fair_penalty
                    full_loss.backward(retain_graph=False)
                    optimizerD.step()

                if constant != 0:
                    gender_correct, occupation_correct, age_correct, random_correct = 0, 0, 0, 0
                    correct = 0
                    for fairD_disc in masked_fairD_set:
                        if fairD_disc is not None:
                            ''' No Gradients Past Here '''
                            with torch.no_grad():
                                task_loss, preds, lhs_emb, rhs_emb = modelD(p_batch_var, \
                                                                            return_embeds=True,
                                                                            filters=masked_filter_set)
                                p_lhs_emb = lhs_emb[:len(p_batch)]
                                filter_emb = p_lhs_emb
                                probs, l_A_labels, l_preds = fairD_disc.predict(filter_emb, p_batch[:, 0], True)
                                l_correct = l_preds.eq(l_A_labels.view_as(l_preds)).sum().item()
                                if fairD_disc.attribute == 'gender':
                                    fairD_gender_loss = fairD_loss.detach().cpu().numpy()
                                    gender_correct += l_correct  #
                                elif fairD_disc.attribute == 'occupation':
                                    fairD_occupation_loss = fairD_loss.detach().cpu().numpy()
                                    occupation_correct += l_correct
                                elif fairD_disc.attribute == 'age':
                                    fairD_age_loss = fairD_loss.detach().cpu().numpy()
                                    age_correct += l_correct
                                else:
                                    fairD_random_loss = fairD_loss.detach().cpu().numpy()
                                    random_correct += l_correct

            gc.collect()

        ########################################### Model Evaluation Start ######################################
        with torch.no_grad():

            pre = {}
            label = {}
            test_loader = DataLoader(test_set, batch_size=1024, num_workers=0, collate_fn=collate_fn,
                                     shuffle=False)
            preds_list = []
            rels_list = []
            test_loss_list = []
            ###################### BATCH VALIDATION ################################
            # for p_batch in test_loader:
            for p_batch in test_loader:
                p_batch_var = Variable(p_batch)  # .cuda()
                lhs, rel, rhs = p_batch_var[:, 0], p_batch_var[:, 1], p_batch_var[:, 2]
                test_loss, preds = modelD(p_batch_var, filters=filter_set)
                rel += 1
                rel = rel.float()
                preds_list.append(preds.squeeze())
                rels_list.append(rel.float())
                test_loss_list.append(test_loss)

                for idx, u_id in enumerate(lhs):
                    feature_value = users_features[u_id]
                    if feature_value not in pre:
                        pre[feature_value] = []
                    pre[feature_value].append(preds[idx].item())
                    if feature_value not in label:
                        label[feature_value] = []
                    label[feature_value].append(rel[idx].item())

            total_preds = torch.cat(preds_list)
            total_rels = torch.cat(rels_list)
            test_loss = torch.mean(torch.stack(test_loss_list))
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_loss_epoch = epoch
            rms = torch.sqrt(F.mse_loss(total_preds.squeeze(), total_rels.squeeze()))
            for el in label:
                label[el] = torch.tensor(label[el])
            for el in pre:
                pre[el] = torch.tensor(pre[el])
            rmse_dict = {}
            RMSE_string = ""
            for el in pre:
                rmse_dict[el] = torch.sqrt(F.mse_loss(pre[el].squeeze(), label[el].squeeze()))
                RMSE_string = RMSE_string + f"RMSE_{el}: {rmse_dict[el]},"
            RMSE_string = RMSE_string[:-1]
            coeff_var_rmse = np.std(list(rmse_dict.values()))/np.mean(list(rmse_dict.values()))
            print(
                f"\nEpoch {epoch}:\tRMSE = {float(rms)}")


        ################################# FAIRNESS TESTING ############################
        f1_age = None
        if use_age_attr:
            acc_age, f1_age = test_age(test_fairness_set, modelD, fairD_age, epoch, filter_set)

        write_eval_measures(logdir, [
            f'epoch:{epoch},RMSE:{rms},RMSE_coeff_var:{coeff_var_rmse},F1_age:{f1_age},best_test_loss:{best_test_loss},best_test_loss_epoch:{best_test_loss_epoch}'])


        if epoch > 0:
            if rms < best_RMSE[0]:
                best_RMSE = [rms, epoch]
            if f1_age < best_F1[0]:
                best_F1 = [f1_age, epoch]


        model_path = f'{logdir}/model_epoch{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': modelD.state_dict(),
        }, model_path)

        if use_age_attr:
            model_path = f'{logdir}/age_model_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': fairD_age.state_dict(),
            }, model_path)

        epoch += 1

    with open(os.path.join(logdir, 'best_epochs.txt'), 'w') as file:
        file.write(f'RMSE,{best_RMSE[1]}\n')
        file.write(f'F1_age,{best_F1[1]}\n')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int, default=10, help='Embedding dimension (default: 50)')
    parser.add_argument('--use_cross_entropy', action='store_true', help="DemPar Discriminators Loss as CE")
    parser.add_argument('--use_age_attr', action='store_true', help='Use Only Age Attribute')
    parser.add_argument('--dataset', type=str, default='bookCrossing', help='')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.001)')
    parser.add_argument('--data_dir', type=str,
                        default=r"datasets")  # ToDO
    parser.add_argument('--eval_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--num_random_items', type=int, default=1000)
    parser.add_argument('--gamma', type=int, default=0, help='Tradeoff for Adversarial Penalty')
    parser.add_argument('--D_steps', type=int, default=10, help='Number of D steps')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='random seed if set == 0. Defined seed if set != 0')

    args = parser.parse_args()

    if args.seed == 0:
        args.seed = random.randint(00000, 99999)

    args.use_age_attr = True

    ########################## DATA PREPARATION ######################
    if args.dataset == "bookCrossing":
        remove_implicit_ratings_BookCrossing(args.data_dir, args.dataset, min_ratings_per_user=20)

    id_remapping_users(args.data_dir, args.dataset, user_train_eval_split=True)
    id_remapping_items(args.data_dir, args.dataset)
    split_rating_data(args.data_dir, args.dataset, args.eval_ratio, args.test_ratio)
    return args


if __name__ == '__main__':
    args = parse_args()
    run_FlexibleFairness(args)




