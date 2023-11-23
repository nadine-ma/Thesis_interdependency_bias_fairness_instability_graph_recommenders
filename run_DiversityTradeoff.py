import copy
import itertools
import os
import random
from argparse import ArgumentParser
import datetime
import ast
import torch.nn.functional as F
import numpy as np
import torch
from sklearn import metrics
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.utils_data_preparation import id_remapping_users, split_rating_data, \
    id_remapping_items, load_user_features, remove_implicit_ratings_BookCrossing
from data_utils.utils import generate_user_item_dicts, generate_rating_list, \
    load_random_items, write_eval_measures
from models.Tradeoff.dataset import RatingsDataset
from models.Tradeoff.GCN import GCN
from models.Tradeoff.optimization_utils import open_graphs
from models.Tradeoff.utils import check_base_files, train_test_split, get_graph_types, float2tensor
from models.metrics import  rmse, average_recommendation_popularity_single_user, get_popularity_distribution
from models.utils import create_complete_user_item_matrix, LF_model, create_uim






def run_DiversityTradeoff(hparams, instance_bool=False, alpha_bool=False):
    """
        Function to run the DiversityTradeoff model

        Parameters:
            hparams : Argparse : Model parameters
            instance_bool : Bool : Whether the model is a seed variation
            alpha_bool : Bool : Whether the model is a alpha variation
        """

    print("Initializing DiversityTradeoff Model...")
    data_dir = hparams.data_dir
    dataset_name = hparams.dataset
    num_random_items = hparams.num_random_items
    batch_size = hparams.batch_size
    top_k = hparams.top_k
    eval_ratio = hparams.eval_ratio
    test_ratio = hparams.test_ratio
    lr = hparams.lr
    graph_type = hparams.graph_type
    sim_order = hparams.sim_order
    dis_order = hparams.dis_order
    num_epochs = hparams.num_epochs
    alpha = hparams.alpha
    mu = hparams.mu
    features = ast.literal_eval(hparams.features)
    training_seed = hparams.seed
    sim_users_NN = hparams.sim_users_NN
    sim_items_NN = hparams.sim_items_NN
    dis_NN = hparams.dis_NN

    ######### SETTING TRAINING SEED ##########
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)

    logdir = os.path.join("logs", dataset_name, "DiversityTradeoff", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(logdir)
    with open(os.path.join(logdir, "hparams.txt"), "w") as hparams_log_file:
        for el in hparams._get_kwargs():
            hparams_log_file.write(f'{el[0]}:{el[1]}:{type(el[1]).__name__}\n')

    if instance_bool is not False:
        with open(os.path.join("logs", dataset_name, "DiversityTradeoff", f'instances_{instance_bool}.txt'), 'a') as file:
            file.write(f'{logdir}\n')

    if alpha_bool is not False:
        with open(os.path.join("logs", dataset_name, "DiversityTradeoff", f'alphas_{alpha_bool}.txt'), 'a') as file:
            file.write(f'{logdir},{alpha}\n')

    users_features = load_user_features(data_dir, dataset_name)
    user_dict_train, item_dict_train, _ = generate_user_item_dicts(data_dir, dataset_name, eval_ratio, test_ratio, train=True)
    uim_full = create_complete_user_item_matrix(data_dir, dataset_name)
    user_num = uim_full.shape[0]
    item_num = uim_full.shape[1]
    if num_random_items == 0:
        num_random_items = item_num
    uim_train = create_uim(data_dir, dataset_name, eval_ratio, test_ratio, train=True)
    uim_eval = create_uim(data_dir, dataset_name, eval_ratio, test_ratio, train=False, eval=True)
    #uim_test = create_uim(data_dir, dataset_name, eval_ratio, test_ratio, train=False, test=True)
    item_LF = LF_model(data_dir, dataset_name, uim_train, eval_ratio, test_ratio)


    item_popularity_distribution = get_popularity_distribution(uim_train)
    item_popularity_distribution_desc_order = list(sorted(item_popularity_distribution, key=lambda item: item[1], reverse=True))

    value_unfairness_items = load_random_items(data_dir, dataset_name, item_num, num_random_items)

    train_set = generate_rating_list(uim_train)
    eval_set = generate_rating_list(uim_eval)
    print(f"{len(train_set)} training samples,  {len(eval_set)} evaluation samples")

    if not check_base_files(data_dir, dataset_name):
        train_test_split(data_dir, dataset_name, uim_train, eval_ratio, test_ratio)

    path = os.path.join(data_dir, dataset_name, 'tradeoff', 'matrices')
    B = np.load(os.path.join(path, f'B_{1-eval_ratio-test_ratio}.npy'))
    C = np.load(os.path.join(path, f'C_{1-eval_ratio-test_ratio}.npy'))

    #### ARCHITECTURE ####
    means = np.mean(uim_train, axis=0)

    graph1, graph2 = get_graph_types(graph_type)
    if graph1 == 'user':
        corr_mat = B
        dict_to_use = item_dict_train
        method1 = 'USERS'
        k = sim_users_NN
    else:
        corr_mat = C
        dict_to_use = user_dict_train
        method1 = 'ITEMS'
        k = sim_items_NN
    GSOSimDict = open_graphs(data_dir, dataset_name, eval_ratio, test_ratio, uim_train, k, corr_mat, dict_to_use, method1, 'similarity')

    if graph2 == 'user':
        corr_mat = B
        dict_to_use = item_dict_train
        method2 = 'USERS'
    else:
        corr_mat = C
        dict_to_use = user_dict_train
        method2 = 'ITEMS'
    k = dis_NN
    GSODisDict = open_graphs(data_dir, dataset_name, eval_ratio, test_ratio, uim_train, k, corr_mat, dict_to_use, method2, 'dissimilarity')

    train_set = RatingsDataset(train_set)
    eval_set = RatingsDataset(eval_set)

    train_dataloader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
    )


    eval_dataloader = DataLoader(
        eval_set,
        batch_size=1024,
        shuffle=False
    )
    model = GCN(graph_type,
                uim_train,
                GSOSimDict,
                GSODisDict,
                sim_order,
                dis_order,
                features,
                alpha)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if not os.path.isdir(f'{data_dir}/{dataset_name}/tradeoff/experiments'):
        os.mkdir(f'{data_dir}/{dataset_name}/tradeoff/experiments')


    best_RMSE = [100, 0]
    best_AD = [0, 0]
    best_ID = [0, 0]



    cnt_batch = 0

    epoch = 0

    while epoch <= num_epochs:
        print()
        ###################### TRAINING ########################
        if epoch > 0:
            model.train()
            optimizer.zero_grad()
            cnt = 0

            batchLoss = 0

            for batch in tqdm(train_dataloader, position=0, leave=True, desc=f'EPOCH {epoch} TRAINING'):

                user = int(batch['user'][0])
                rating = int(batch['rating'][0])
                item = int(batch['item'][0])
                yHat = model(user, item)

                cnt_batch += 1

                rating = rating - means[item]
                reg = mu * model.l2regularizer(application='GCN')
                loss = criterion(yHat[0], torch.squeeze(float2tensor(rating))) + reg
                loss.backward()
                batchLoss += loss

                if cnt_batch == batch_size:
                    optimizer.step()
                    optimizer.zero_grad()
                    cnt_batch = 0
                    batchLoss = 0

                cnt = cnt + 1

        ########################################### Model Evaluation Start ######################################
        model.eval()
        uim_graph_predicted = model.prediction(alpha=alpha, verbose=False)
        ###################### BATCH VALIDATION ################################
        rmse_all = []


        # for batch in tqdm(eval_dataloader, position=0, leave=True, desc='BATCH VALIDATION EPOCH ' + str(epoch)):
        for batch in tqdm(eval_dataloader, position=0, leave=True, desc=f'EPOCH {epoch} BATCH EVALUATION'):
            user = batch['user']
            rating = batch['rating']
            item = batch['item']

            ################################################################################################
            with torch.no_grad():
                user_val = int(user[0])
                rating_val = int(rating[0])
                item_val = int(item[0])

                reg = mu * model.l2regularizer()
                yHatVal = model(user_val, item_val)
                rating_val = rating_val - means[item_val]
                lossVal = criterion(yHatVal[0], float2tensor(rating_val).squeeze()) + reg
            ###################################################################################################

            pre_all = []
            pre_features = {}
            label_features = {}
            label_all = []
            i = 0
            while i < len(user):
                u_id = user[i].item()
                r_v = rating[i].item()
                i_id = item[i]
                pre_get = uim_graph_predicted[u_id, i_id]
                pre_all.append(pre_get)
                label_all.append(r_v)
                feature = users_features[u_id]  # [1].item()
                if feature not in pre_features:
                    pre_features[feature] = []
                pre_features[feature].append(pre_get)
                if feature not in label_features:
                    label_features[feature] = []
                label_features[feature].append(r_v)
                i += 1

            rmse_all.append(rmse(np.array(pre_all), label_all))

        rmse_all = round(np.nanmean(rmse_all), 4)

        for el in label_features:
            label_features[el] = torch.tensor(label_features[el])
        for el in pre_features:
            pre_features[el] = torch.tensor(pre_features[el])
        rmse_dict = {}
        RMSE_string = ""
        for el in pre_features:
            rmse_dict[el] = torch.sqrt(F.mse_loss(pre_features[el].squeeze(), label_features[el].squeeze()))
            RMSE_string = RMSE_string + f"RMSE_{el}:{rmse_dict[el]},"

        RMSE_string = RMSE_string[:-1]
        coeff_var_rmse = np.std(list(rmse_dict.values()))/np.mean(list(rmse_dict.values()))

        print(f'Epoch {epoch}: RMSE={rmse_all}')

        ####################### RANKING VALIDATION ##########################

        users_diversity = []
        users_parity_date = []
        users_parity_genre = []
        recommendations = []
        value_unfairness_items_av_prediction = {}
        value_unfairness_items_av_rating = {}
        average_recommendation_popularity = 0


        for u_id in tqdm(range(user_num), position=0, leave=True, desc=f'EPOCH {epoch} RANKING VALIDATION'):
            user_top_k = []
            user_all_pre = {}  ####
            for idx in range(item_num):  ####
                user_all_pre[idx] = uim_graph_predicted[u_id, idx]

            # Value Unfairness
            feature = users_features[u_id]
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
            if user_div:  # some users only have one item, there is no dissimilarity computable, thus these get skipped
                users_diversity.append(
                    np.mean(np.array(user_div)))  # calculate average (mean) diversity for every users

            average_recommendation_popularity += average_recommendation_popularity_single_user(
                item_popularity_distribution, user_top_k)

        unique = list(dict.fromkeys(recommendations))
        AD = round((len(unique) / item_num), 4)  # uim_predicted.shape[1]=number of all items
        ID = round(np.mean(np.array(users_diversity)), 4)  # mean all individual users diversity values
        average_recommendation_popularity = round((average_recommendation_popularity / user_num), 4)
        arp_divided_by_most_rated = round(
            (average_recommendation_popularity / item_popularity_distribution_desc_order[0][1]), 4)

        ##########################################

        # Value Unfairness
        mean_ratings = {}
        mean_predictions = {}
        for feature in value_unfairness_items_av_rating:
            if feature not in mean_predictions:
                mean_predictions[feature] = []
            if feature not in mean_ratings:
                mean_ratings[feature] = []
            for item in value_unfairness_items_av_rating[feature]:
                mean_rating = np.mean(value_unfairness_items_av_rating[feature][item])
                mean_ratings[feature].append(mean_rating)
            for item in value_unfairness_items_av_prediction[feature]:
                mean_prediction = np.mean(value_unfairness_items_av_prediction[feature][item])
                mean_predictions[feature].append(mean_prediction)

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
        VU_string = VU_string [:-1]
        coeff_var_vu = np.std(list(vu_dict.values())) / np.mean(list(vu_dict.values()))

        write_eval_measures(logdir, [
            f'epoch:{epoch},RMSE:{rmse_all},AD@{top_k}:{AD},ID@{top_k}:{ID},ARP@{top_k}:{arp_divided_by_most_rated},RMSE_coeff_var:{coeff_var_rmse},VU_coeff_var:{coeff_var_vu},{RMSE_string},{VU_string}'])

        if epoch > 0:
            if rmse_all < best_RMSE[0]:
                best_RMSE = [rmse_all, epoch]
            if AD > best_AD[0]:
                best_AD = [AD, epoch]
            if ID > best_ID[0]:
                best_ID = [ID, epoch]

        model_path = f'{logdir}/model_epoch{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, model_path)

        epoch += 1


    with open(os.path.join(logdir, 'best_epochs.txt'), 'w') as file:
        file.write(f'RMSE,{best_RMSE[1]}\n')
        file.write(f'AD@{top_k},{best_AD[1]}\n')
        file.write(f'ID@{top_k},{best_ID[1]}\n')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"datasets")
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--num_random_items", type=int, default=1000)  # 0 = all items
    parser.add_argument("--batch_size", type=int, default=128)  # 1 #256
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)

    parser.add_argument("--graph_type", type=str, default='IU')
    parser.add_argument("--sim_order", type=int, default=1)
    parser.add_argument("--dis_order", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--features", type=str, default='[1,2]')
    parser.add_argument("--sim_users_NN", type=int, default=30)
    parser.add_argument("--sim_items_NN", type=int, default=35)
    parser.add_argument("--dis_NN", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0, help='0 = random seed')
    args = parser.parse_args()

    if args.seed == 0:
        args.seed = random.randint(00000, 99999)

    ########################## DATA PREPARATION ######################

    if args.dataset == "bookCrossing":
        remove_implicit_ratings_BookCrossing(args.data_dir, args.dataset, min_ratings_per_user=10)

    id_remapping_users(args.data_dir, args.dataset, user_train_eval_split=True)
    id_remapping_items(args.data_dir, args.dataset)
    split_rating_data(args.data_dir, args.dataset, args.eval_ratio, args.test_ratio)
    return args



if __name__ == '__main__':
    args = parse_args()
    run_DiversityTradeoff(args)






