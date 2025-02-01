import csv
import itertools
import os
from argparse import ArgumentParser

from torch.autograd import Variable
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from data_utils.utils import write_eval_measures, generate_user_list, load_random_items, generate_rating_list
from data_utils.utils_data_preparation import load_user_features
from models.FlexFair.model import SharedBilinearDecoder, SimpleGCMC, AgeDiscriminator
from models.FlexFair.transD_FlexFair import collate_fn, KBDataset, optimizer
from models.FlexFair.utils import NodeClassification
from models.FlexFair.eval_FlexFair import test_age
from models.metrics import average_recommendation_popularity_single_user, get_popularity_distribution
from models.utils import LF_model, create_uim, create_complete_user_item_matrix



def test_checkpoint_FlexibleFairness(run_dir, eval_measure, epoch, instance_bool=False, gamma_bool=False):

    parser = ArgumentParser()
    args = parser.parse_args()

    with open(os.path.join(run_dir, 'hparams.txt')) as f:
        for line in f:
            x = line.strip('\n').split(":")
            value = x[1]
            if x[2] == 'float':
                value = float(value)
            elif x[2] == 'int':
                value = int(value)
            elif x[2] == 'bool':
                if value == 'False':
                    value = False
                else:
                    value = True
            setattr(args, x[0], value)

    data_dir = args.data_dir
    dataset_name = args.dataset
    num_random_items = args.num_random_items
    batch_size = args.batch_size
    top_k = args.top_k
    eval_ratio = args.eval_ratio
    test_ratio = args.test_ratio
    lr = args.lr
    d_steps = args.D_steps
    embed_dim = args.embed_dim
    gamma = args.gamma
    num_epochs = args.num_epochs
    use_age_attr = args.use_age_attr
    use_cross_entropy = args.use_cross_entropy
    training_seed = args.seed

    device = "cpu"

    checkpoint_dir_model = os.path.join(run_dir, f'model_epoch{epoch}.pth')
    checkpoint_dir_age_class = os.path.join(run_dir, f'age_model_epoch{epoch}.pth')


    uim_full = create_complete_user_item_matrix(data_dir, dataset_name)
    num_users = uim_full.shape[0]
    num_movies = uim_full.shape[1]
    uim_train = create_uim(data_dir, dataset_name, eval_ratio, test_ratio, train=True, eval=False)
    uim_test = create_uim(data_dir, dataset_name, eval_ratio, test_ratio, train=False, test=True)

    if dataset_name != "bookCrossing":
        item_LF = LF_model(data_dir, dataset_name, uim_train, eval_ratio, test_ratio)

    item_popularity_distribution = get_popularity_distribution(uim_train)
    item_popularity_distribution_desc_order = list(
        sorted(item_popularity_distribution, key=lambda item: item[1], reverse=True))

    value_unfairness_items = load_random_items(data_dir, dataset_name, num_movies, 1000)
    users_features = load_user_features(data_dir, dataset_name)

    test_ratings = generate_rating_list(uim_test)

    num_ent = num_users + num_movies

    num_rel = 5
    if "bookCrossing" in run_dir:
        num_rel = 10
    users_test = generate_user_list(data_dir, dataset_name, train=False, eval=True)
    users_train = generate_user_list(data_dir, dataset_name, train=True, eval=False)
    train_fairness_set = NodeClassification(users_train)

    test_set = KBDataset(test_ratings, num_users)
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
        checkpoint_age_classifier = torch.load(checkpoint_dir_age_class)
        fairD_age.load_state_dict(checkpoint_age_classifier['model_state_dict'])
        optimizer_fairD_age = optimizer(fairD_age.parameters(), 'adam', lr)


    ''' Create Sets '''
    filter_set = [gender_filter, occupation_filter, age_filter, None]

    checkpoint_model = torch.load(checkpoint_dir_model)
    modelD.load_state_dict(checkpoint_model['model_state_dict'])

    with torch.no_grad():
        pre = {}
        label = {}
        test_loader = DataLoader(test_set, batch_size=1024, num_workers=0, collate_fn=collate_fn,
                                 shuffle=False)
        preds_list = []
        rels_list = []
        ###################### BATCH VALIDATION ################################

        for p_batch in tqdm(test_loader, position=0, leave=True, desc='BATCH VALIDATION'):
            p_batch_var = Variable(p_batch)  # .cuda()
            lhs, rel, rhs = p_batch_var[:, 0], p_batch_var[:, 1], p_batch_var[:, 2]
            test_loss, preds = modelD(p_batch_var, filters=filter_set)
            rel += 1
            rel = rel.float()
            preds_list.append(preds.squeeze())
            rels_list.append(rel.float())

            for idx, u_id in enumerate(lhs):
                feature_value = users_features[u_id]
                if feature_value not in pre:
                    pre[feature_value] = []
                pre[feature_value].append(preds[idx].item())
                if feature_value not in label:
                    label[feature_value] = []
                label[feature_value].append(rel[idx].item())

        preds_all = torch.cat(preds_list)
        total_rels = torch.cat(rels_list)
        rms = torch.sqrt(F.mse_loss(preds_all.squeeze(), total_rels.squeeze()))
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
        coeff_var_rmse = np.std(list(rmse_dict.values())) / np.mean(list(rmse_dict.values()))



        ####################### RANKING VALIDATION ##########################

        recommendations = []
        users_diversity = []
        value_unfairness_items_av_prediction = {}
        value_unfairness_items_av_rating = {}
        average_recommendation_popularity = 0

        modelD.eval()
        for u_id in tqdm(range(num_users), position=0, leave=True, desc='RANKING VALIDATION'):
            feature_value = users_features[u_id]
            user_top_k = []
            user_all_pre = {}  ####
            user_pre = [i[0] for i in modelD.predict_uim(u_id, num_users,
                                                         num_movies).tolist()]

            for idx, element in enumerate(user_pre):
                user_all_pre[idx] = element

            # Value Unfairness
            for el in value_unfairness_items:
                if feature_value not in value_unfairness_items_av_prediction:
                    value_unfairness_items_av_prediction[feature_value] = {}
                if el not in value_unfairness_items_av_prediction[feature_value]:
                    value_unfairness_items_av_prediction[feature_value][el] = []
                value_unfairness_items_av_prediction[feature_value][el].append(
                    user_all_pre[el])
                if feature_value not in value_unfairness_items_av_rating:
                    value_unfairness_items_av_rating[feature_value] = {}
                if el not in value_unfairness_items_av_rating[feature_value]:
                    value_unfairness_items_av_rating[feature_value][el] = []
                value_unfairness_items_av_rating[feature_value][el].append(
                    uim_full[u_id][el])


            # Get Top K items for user
            pre_user = dict(
                sorted(user_all_pre.items(), key=lambda item: item[1], reverse=True))  ####

            for idx, key in enumerate(pre_user):
                user_top_k.append(key)
                recommendations.append(key)
                if idx == top_k - 1:
                    break

            # ID
            if dataset_name != "bookCrossing":
                user_div = []
                for i1, i2 in itertools.combinations(user_top_k, 2):
                    # goes through all possible item combinations (combi=2) in top_k_recom.
                    dissimilarity = item_LF[i1, i2]
                    user_div.append(dissimilarity)
                if user_div:  #ToDO: if entfernen?
                    users_diversity.append(
                        np.mean(
                            np.array(user_div)))  # calculate average (mean) diversity for every users
                else:
                    pass

            average_recommendation_popularity += average_recommendation_popularity_single_user(
                item_popularity_distribution, user_top_k)

        unique = list(dict.fromkeys(recommendations))
        AD = round((len(unique) / num_movies), 4)
        if dataset_name != "bookCrossing":
            ID = round(np.mean(np.array(users_diversity)), 4)
        else:
            ID = None
        average_recommendation_popularity = round((average_recommendation_popularity / num_users), 4)
        arp_divided_by_most_rated = round(
            (average_recommendation_popularity / item_popularity_distribution_desc_order[0][1]), 4)

        ################# GINI ###################
        top_k_rec = {}
        for el in recommendations:
            if el not in top_k_rec:
                top_k_rec[el] = 0
            top_k_rec[el] += 1
        for el in range(num_movies):
            if el not in top_k_rec:
                top_k_rec[el] = 0

        gini_top = 0
        gini_bottom = 0
        for i1 in top_k_rec:
            for i2 in top_k_rec:
                gini_top += abs(top_k_rec[i1] - top_k_rec[i2])
                gini_bottom += top_k_rec[i2]

        gini = round((gini_top / (2 * gini_bottom)), 4)


        #################### Value Unfairness ############################
        mean_ratings = {}
        mean_predictions = {}
        mean_all = {}
        for gender in value_unfairness_items_av_rating:
            if gender not in mean_predictions:
                mean_predictions[gender] = []
            if gender not in mean_ratings:
                mean_ratings[gender] = []
            if gender not in mean_all:
                mean_all[gender] = []
            for item in value_unfairness_items_av_rating[gender]:
                mean_rating = np.mean(value_unfairness_items_av_rating[gender][item])
                mean_ratings[gender].append(mean_rating)
                mean_all[gender] += np.subtract(value_unfairness_items_av_prediction[gender][item],
                                                value_unfairness_items_av_rating[gender][item]).tolist()
                mean_prediction = np.mean(value_unfairness_items_av_prediction[gender][item])
                mean_predictions[gender].append(mean_prediction)

        VU_string1 = ""
        for gender in mean_all:
            mean_value = np.mean(mean_all[gender])
            VU_string1 += f"VU1_{gender}: {mean_value},"

        value_unfairness_rating = {}
        for idx, el in enumerate(value_unfairness_items):
            if el not in value_unfairness_rating:
                value_unfairness_rating[el] = {}
            for feat in mean_predictions:
                if feat not in value_unfairness_rating[el]:
                    value_unfairness_rating[el][feat] = []
                value_unfairness_rating[el][feat] = (abs(mean_predictions[feat][idx] - mean_ratings[feat][idx]))

        vu_dict = {}
        VU_string = ""
        for feat in mean_predictions:
            vu_dict[feat] = sum(
                [value_unfairness_rating[el][feat] for el in value_unfairness_rating]) / len(value_unfairness_items)
            VU_string = VU_string + f"VU_{feat}:{vu_dict[feat]},"
        VU_string = VU_string[:-1]
        coeff_var_vu = np.std(list(vu_dict.values())) / np.mean(list(vu_dict.values()))
    modelD.train()


    ################################# FAIRNESS TESTING ############################
    f1_age = None
    if use_age_attr:
        acc_age,f1_age = test_age(test_fairness_set, modelD, fairD_age, epoch, filter_set)


    write_eval_measures(run_dir, [
        f'epoch:{epoch},RMSE:{rms},AD@{top_k}:{AD},ID@{top_k}:{ID},ARP@{top_k}:{arp_divided_by_most_rated},Gini@{top_k}:{gini},F1_age:{f1_age},RMSE_coeff_var:{coeff_var_rmse},VU_coeff_var:{coeff_var_vu},{RMSE_string},{VU_string}'],
                        test=True)

    if instance_bool is not False:
        with open(os.path.join(run_dir, '..', f'instances_{eval_measure}_tested.txt'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                f'run_dir:{run_dir},epoch:{epoch},RMSE:{rms},AD@{top_k}:{AD},ID@{top_k}:{ID},ARP@{top_k}:{arp_divided_by_most_rated},F1_age:{f1_age},'
                f'RMSE_coeff_var:{coeff_var_rmse},VU_coeff_var:{coeff_var_vu},{RMSE_string},{VU_string}'])

    if gamma_bool is not False:
        with open(os.path.join(run_dir, '..', f'gammas_{eval_measure}_tested.txt'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                f'gamma:{gamma},run_dir:{run_dir},epoch:{epoch},RMSE:{rms},AD@{top_k}:{AD},ID@{top_k}:{ID},ARP@{top_k}:{arp_divided_by_most_rated},F1_age:{f1_age},'
                f'RMSE_coeff_var:{coeff_var_rmse},VU_coeff_var:{coeff_var_vu},{RMSE_string},{VU_string}'])


    return np.array(preds_all)


