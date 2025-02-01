
import csv
import itertools
import os
from argparse import ArgumentParser
import ast
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.utils import get_total_user_item_counts, \
    generate_user_item_dicts, generate_rating_list, \
    load_random_items, write_eval_measures
from data_utils.utils_data_preparation import load_user_features
from models.Tradeoff.dataset import RatingsDataset
from models.Tradeoff.GCN import GCN
from models.Tradeoff.optimization_utils import open_graphs
from models.Tradeoff.utils import get_graph_types
from models.metrics import rmse, average_recommendation_popularity_single_user, get_popularity_distribution
from models.utils import create_complete_user_item_matrix, LF_model, create_uim


def test_checkpoint_DiversityTradeoff(run_dir, eval_measure, epoch, instance_bool=False, alpha_bool=False):
    """
    Function to test the recommendation objectives produced by a _DiversityTradeoff model checkpoint

    Parameters:
        run_dir : str : path to model folder
        eval_measure : str : info about the optimisation objective of the model to test. Needed for result allocation
        epoch : int : model epoch to test
        instance_bool : Bool : Whether the model is a seed variation
        alpha_bool : Bool : Whether the model is an alpha variation

    Returns:
        pred_all : np.array : Contains the rating predictions for all test samples
    """

    parser = ArgumentParser()
    hparams = parser.parse_args()

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
            setattr(hparams, x[0], value)

    data_dir = hparams.data_dir
    dataset = hparams.dataset
    num_random_items = hparams.num_random_items
    top_k = hparams.top_k
    graph_type = hparams.graph_type
    sim_order = hparams.sim_order
    dis_order = hparams.dis_order
    alpha = hparams.alpha
    mu = hparams.mu
    features = ast.literal_eval(hparams.features)
    eval_ratio = hparams.eval_ratio
    test_ratio = hparams.test_ratio
    # run_dir = hparams.run_dir

    checkpoint_dir = os.path.join(run_dir, f'model_epoch{epoch}.pth')

    user_num, item_num = get_total_user_item_counts(data_dir, dataset)
    users_features = load_user_features(data_dir, dataset)
    user_dict_train, item_dict_train, _ = generate_user_item_dicts(data_dir, dataset, eval_ratio, test_ratio,
                                                                   train=True)
    uim_full = create_complete_user_item_matrix(data_dir, dataset)
    uim_train = create_uim(data_dir, dataset, eval_ratio, test_ratio, train=True)
    uim_test = create_uim(data_dir, dataset, eval_ratio, test_ratio, train=False,
                          test=True)  # TODO: test=True#############################################################
    item_LF = LF_model(data_dir, dataset, uim_train, eval_ratio, test_ratio)

    item_popularity_distribution = get_popularity_distribution(uim_train)
    item_popularity_distribution_desc_order = list(
        sorted(item_popularity_distribution, key=lambda item: item[1], reverse=True))

    value_unfairness_items = load_random_items(data_dir, dataset, item_num, num_random_items)

    path = os.path.join(data_dir, dataset, 'tradeoff', 'matrices')
    B = np.load(os.path.join(path, f'B_{1 - eval_ratio - test_ratio}.npy'))
    C = np.load(os.path.join(path, f'C_{1 - eval_ratio - test_ratio}.npy'))

    #### ARCHITECTURE ####
    means = np.mean(uim_train, axis=0)

    graph1, graph2 = get_graph_types(graph_type)
    if graph1 == 'user':
        corr_mat = B
        dict_to_use = item_dict_train
        method1 = 'USERS'
        k = 30
    else:
        corr_mat = C
        dict_to_use = user_dict_train
        method1 = 'ITEMS'
        k = 35
    GSOSimDict = open_graphs(data_dir, dataset, eval_ratio, test_ratio, uim_train, k, corr_mat, dict_to_use, method1,
                             'similarity')

    if graph2 == 'user':
        corr_mat = B
        dict_to_use = item_dict_train
        method2 = 'USERS'
    else:
        corr_mat = C
        dict_to_use = user_dict_train
        method2 = 'ITEMS'
    k = 40
    GSODisDict = open_graphs(data_dir, dataset, eval_ratio, test_ratio, uim_train, k, corr_mat, dict_to_use, method2,
                             'dissimilarity')

    model = GCN(graph_type,
                uim_train,
                GSOSimDict,
                GSODisDict,
                sim_order,
                dis_order,
                features,
                alpha)

    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_set = generate_rating_list(uim_test)
    test_set = RatingsDataset(test_set)

    test_dataloader = DataLoader(
        test_set,
        batch_size=2048,
        shuffle=False)

    model.eval()
    uim_graph_predicted = model.prediction(alpha=alpha, verbose=False)
    ###################### BATCH VALIDATION ################################
    rmse_all = []
    rmse_gender0 = []
    rmse_gender1 = []
    for batch in tqdm(test_dataloader, position=0, leave=True, desc='BATCH VALIDATION'):
        user = batch['user']
        rating = batch['rating']
        item = batch['item']
        pre_all = []
        pre_genders = {}
        label_genders = {}
        label_all = []
        i = 0
        while i < len(user):
            u_id = user[i].item()
            r_v = rating[i].item()
            i_id = item[i]
            pre_get = uim_graph_predicted[u_id, i_id]
            pre_all.append(pre_get)
            label_all.append(r_v)

            feature = users_features[u_id]

            if feature not in pre_genders:  #########
                pre_genders[feature] = []  #########
            pre_genders[feature].append(pre_get)  #########
            if feature not in label_genders:  #########
                label_genders[feature] = []  #########
            label_genders[feature].append(r_v)  #########
            i += 1

        rmse_all.append(rmse(np.array(pre_all), label_all))

    rmse_all = round(np.nanmean(rmse_all), 4)

    for el in label_genders:
        label_genders[el] = torch.tensor(label_genders[el])
    for el in pre_genders:
        pre_genders[el] = torch.tensor(pre_genders[el])
    rmse_dict = {}
    RMSE_string = ""
    for el in pre_genders:
        rmse_dict[el] = torch.sqrt(F.mse_loss(pre_genders[el].squeeze(), label_genders[el].squeeze()))
        RMSE_string = RMSE_string + f"RMSE_{el}: {rmse_dict[el]},"
    RMSE_string = RMSE_string[:-1]
    coeff_var_rmse = np.std(list(rmse_dict.values())) / np.mean(list(rmse_dict.values()))


    ####################### RANKING VALIDATION ##########################

    users_diversity = []

    recommendations = []
    value_unfairness_items_av_prediction = {}
    value_unfairness_items_av_rating = {}
    average_recommendation_popularity = 0

    for u_id in tqdm(range(user_num), position=0, leave=True, desc='RANKING VALIDATION'):

        feature = users_features[u_id]

        user_top_k = []
        user_all_pre = {}
        for idx in range(item_num):
            user_all_pre[idx] = uim_graph_predicted[u_id, idx]

        # Value Unfairness
        for el in value_unfairness_items:
            if feature not in value_unfairness_items_av_prediction:
                value_unfairness_items_av_prediction[feature] = {}
            if el not in value_unfairness_items_av_prediction[feature]:
                value_unfairness_items_av_prediction[feature][el] = []
            value_unfairness_items_av_prediction[feature][el].append(user_all_pre[el])
            if feature not in value_unfairness_items_av_rating:
                value_unfairness_items_av_rating[feature] = {}
            if el not in value_unfairness_items_av_rating[feature]:
                value_unfairness_items_av_rating[feature][el] = []
            value_unfairness_items_av_rating[feature][el].append(uim_full[u_id][el])

        # Get Top K items for user
        pre_user = dict(sorted(user_all_pre.items(), key=lambda item: item[1], reverse=True))  ####

        for idx, key in enumerate(pre_user):
            user_top_k.append(key)
            recommendations.append(key)
            if idx == top_k - 1:
                break

        ##########  ID
        user_div = []
        for i1, i2 in itertools.combinations(user_top_k, 2):
            # goes through all possible item combinations (combi=2) in top_k_recom.
            dissimilarity = item_LF[i1, i2]
            user_div.append(dissimilarity)
        if user_div:  # ToDo: some users only have one item, there is no dissimilarity computable, thus these get skipped
            users_diversity.append(
                np.mean(np.array(user_div)))  # calculate average (mean) diversity for every users

        average_recommendation_popularity += average_recommendation_popularity_single_user(
            item_popularity_distribution, user_top_k)

    #
    unique = list(dict.fromkeys(recommendations))
    AD = round((len(unique) / item_num), 4)  # uim_predicted.shape[1]=number of all items
    ID = round(np.mean(np.array(users_diversity)), 4)  # mean all individual users diversity values
    average_recommendation_popularity = round((average_recommendation_popularity / user_num), 4)  # / \
    arp_divided_by_most_rated = round(
        (average_recommendation_popularity / item_popularity_distribution_desc_order[0][1]), 4)

    ################# GINI ###################
    top_k_rec = {}
    for el in recommendations:
        if el not in top_k_rec:
            top_k_rec[el] = 0
        top_k_rec[el] += 1
    for el in range(item_num):
        if el not in top_k_rec:
            top_k_rec[el] = 0
    # top_k_rec = {k: v for k, v in sorted(top_k_rec.items(), key=lambda item: item[1], reverse=True)}
    gini_top = 0
    gini_bottom = 0
    for i1 in top_k_rec:
        for i2 in top_k_rec:
            gini_top += abs(top_k_rec[i1] - top_k_rec[i2])
            gini_bottom += top_k_rec[i2]
    gini = round((gini_top / (2 * gini_bottom)), 4)

    # Value Unfairness
    mean_ratings = {}
    mean_predictions = {}
    for gender in value_unfairness_items_av_rating:
        if gender not in mean_predictions:
            mean_predictions[gender] = []
        if gender not in mean_ratings:
            mean_ratings[gender] = []
        for item in value_unfairness_items_av_rating[gender]:
            mean_rating = np.mean(value_unfairness_items_av_rating[gender][item])
            mean_ratings[gender].append(mean_rating)
        for item in value_unfairness_items_av_prediction[gender]:
            mean_prediction = np.mean(value_unfairness_items_av_prediction[gender][item])
            mean_predictions[gender].append(mean_prediction)

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


    write_eval_measures(run_dir,[
            f'epoch:{epoch},RMSE:{rmse_all},AD@{top_k}:{AD},ID@{top_k}:{ID},ARP@{top_k}:{arp_divided_by_most_rated},'
            f'RMSE_coeff_var:{coeff_var_rmse},VU_coeff_var:{coeff_var_vu},{RMSE_string},{VU_string}'],test=True)

    if instance_bool is not False:
        with open(os.path.join(run_dir, '..', f'instances_{eval_measure}_tested.txt'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                f'run_dir:{run_dir},epoch:{epoch},RMSE:{rmse_all},AD@{top_k}:{AD},ID@{top_k}:{ID},ARP@{top_k}:{arp_divided_by_most_rated},'
                f'RMSE_coeff_var:{coeff_var_rmse},VU_coeff_var:{coeff_var_vu},{RMSE_string},{VU_string}'])

    if alpha_bool is not False:
        with open(os.path.join(run_dir, '..', f'alphas_{eval_measure}_tested.txt'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                f'alpha:{alpha},run_dir:{run_dir},epoch:{epoch},RMSE:{rmse_all},AD@{top_k}:{AD},ID@{top_k}:{ID},ARP@{top_k}:{arp_divided_by_most_rated},'
                f'RMSE_coeff_var:{coeff_var_rmse},VU_coeff_var:{coeff_var_vu},{RMSE_string},{VU_string}'])

    return pre_all

