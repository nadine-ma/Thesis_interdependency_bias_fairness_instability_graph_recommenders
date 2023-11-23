import os
import scipy
import numpy as np
from tqdm import tqdm
from data_utils.utils import get_total_user_item_counts

# def generate_user_item_rating(data_dir, dataset, train=True, eval=False, test=False):
#     ratings = []
#     path = ""
#     if test:
#         path = "test"
#     elif eval:
#         path = "eval"
#     elif train:
#         path = "train"
#
#     with open(os.path.join(data_dir, dataset, 'ratings_' + path + '.csv')) as f:
#         for idx, line in enumerate(f):
#             parts = line.strip('\n').split(',')
#             user = int(parts[0])
#             movie_id = int(parts[1])
#             rating = int(parts[2])
#             ratings.append([user, movie_id, rating])  # ToDo: order in list based on rating_dict from FairGo
#
#     ratings = np.array(ratings)
#     print("finished generate_user_item_rating")
#     return ratings


def create_complete_user_item_matrix(data_dir, dataset):
    if not os.path.exists(os.path.join(data_dir, dataset, 'UIM.npy')):
        print("creating full user item matrix")
        ratings = []
        with open(os.path.join(data_dir, dataset, 'ratings_remapped.csv')) as f:
            for line in f:
                parts = line.strip('\n').split(',')
                user = float(parts[0])
                item = float(parts[1])
                rating = float(parts[2])
                ratings.append([user, item, rating])
        ratings = np.array(ratings)

        rows, row_pos = np.unique(ratings[:, 0], return_inverse=True)
        cols, col_pos = np.unique(ratings[:, 1], return_inverse=True)
        # rows, row_pos = np.unique(ratings[:, 0], return_inverse=True)
        # cols, col_pos = np.unique(ratings[:, 1], return_inverse=True)
        uim = np.zeros((len(rows), len(cols)), dtype=ratings.dtype)
        uim[row_pos, col_pos] = ratings[:, 2]
        np.save(os.path.join(data_dir, dataset, 'UIM.npy'), uim)
        print("finished creating full user item matrix")
    else:
        uim = np.load(os.path.join(data_dir, dataset,  'UIM.npy'))
    return uim


def create_uim(data_dir, dataset, eval_ratio, test_ratio, train=True, eval=False, test=False):
    split=f"train_{1-eval_ratio-test_ratio}"
    if eval:
        split=f"eval_{eval_ratio}"
    if test:
        split=f"test_{eval_ratio}"

    if not os.path.exists(os.path.join(data_dir, dataset, f'UIM_{split}.npy')):
        print(f"creating user item matrix")
        user_num, item_num = get_total_user_item_counts(data_dir, dataset)
        ratings = []
        with open(os.path.join(data_dir, dataset, f'ratings_{split}.csv')) as f:
            for line in f:
                parts = line.strip('\n').split(',')
                user = float(parts[0])
                item = float(parts[1])
                rating = float(parts[2])
                ratings.append([user, item, rating])
        ratings = np.array(ratings)

        uim = np.zeros((user_num, item_num), dtype=ratings.dtype)
        for el in ratings:
            uim[int(el[0]), int(el[1])] = el[2]
        np.save(os.path.join(data_dir, dataset, f'UIM_{split}.npy'), uim)
        print("finished creating user item matrix")
    else:
        uim = np.load(os.path.join(data_dir, dataset, f'UIM_{split}.npy'))
    return uim

def create_uim_eval_binary(data_dir, dataset):
    user_num, item_num = get_total_user_item_counts(data_dir, dataset)
    ratings = []
    if not os.path.exists(os.path.join(data_dir, dataset, 'UIM_eval_binary.npy')):
        with open(os.path.join(data_dir, dataset, 'ratings_eval.csv')) as f:
            for line in f:
                parts = line.strip('\n').split(',')
                user = float(parts[0])
                item = float(parts[1])
                rating = float(parts[2])
                if rating > 0:
                    rating = 1
                ratings.append([user, item, rating])
        ratings = np.array(ratings)

        uim = np.zeros((user_num, item_num), dtype=ratings.dtype)
        for el in ratings:
            uim[int(el[0]), int(el[1])] = el[2]
        np.save(os.path.join(data_dir, dataset, 'UIM_eval_binary.npy'), uim)
    else:
        uim = np.load(os.path.join(data_dir, dataset, 'UIM_eval_binary.npy'))
    return uim



def LF_model(data_dir, dataset, uim_train, eval_ratio, test_ratio, num_factors=7):
    if not os.path.exists(os.path.join(data_dir, dataset, f'item_LF_{1-eval_ratio-test_ratio}.npy')):
        print("LF model")
        uim_train = np.nan_to_num(uim_train)
        u, s, vh = np.linalg.svd(uim_train)
        k = num_factors
        vh = vh[0:k, :]
        item_sim = np.zeros(shape=(uim_train.shape[1], uim_train.shape[1]))
        for i in tqdm(range(0, uim_train.shape[1]), position=0, leave=True, desc='Item LF'):
            for j in range(i + 1, uim_train.shape[1]):
                #dist = np.linalg.norm(vh[:, i] - vh[:, j])   #identical to: np.sum(np.abs(vh[:,i]-vh[:,j])**2,axis=(0))**(1./2)
                dist = np.sum(np.abs(vh[:,i]-vh[:,j])**2,axis=(0))**(1./2)   #ToDo
                item_sim[i, j] = dist
                item_sim[j, i] = dist
        item_LF = np.nan_to_num(item_sim)
        np.save(os.path.join(data_dir, dataset, f'item_LF_{1-eval_ratio-test_ratio}.npy'), item_LF)
        print("finished item LF creation")
    else:
        item_LF = np.load(os.path.join(data_dir, dataset, f'item_LF_{1-eval_ratio-test_ratio}.npy'))
    return item_LF

#####################
def LF_model_sparse(data_dir, dataset, uim_train, eval_ratio, test_ratio, num_factors=7):
    if not os.path.exists(os.path.join(data_dir, dataset, f'item_LF_{1-eval_ratio-test_ratio}.npz')):
        #uim_train = np.nan_to_num(uim_train)
        # u, s, vh = scipy.linalg.svd(uim_train)
        u, s, vh = scipy.sparse.linalg.svds(uim_train)
        k = num_factors
        vh = vh[0:k, :]
        item_sim = scipy.sparse.lil_matrix((uim_train.shape[1], uim_train.shape[1]))
        #item_sim = np.zeros(shape=(uim_train.shape[1], uim_train.shape[1]))
        for i in tqdm(range(0, uim_train.shape[1]), position=0, leave=True, desc='Item LF'):
            for j in range(i + 1, uim_train.shape[1]):
                #dist = scipy.linalg.norm(vh[:, i] - vh[:, j]) #################
                dist = np.sum(np.abs(vh[:, i] - vh[:, j]) ** 2, axis=(0)) ** (1. / 2)
                item_sim[i, j] = dist
                item_sim[j, i] = dist
        item_sim = scipy.sparse.csr_matrix(item_sim)
        scipy.sparse.save_npz(os.path.join(data_dir, dataset,f'item_LF_{1-eval_ratio-test_ratio}.npz'), item_sim)
        print("finished item LF creation")
    else:
        item_sim = scipy.sparse.load_npz(os.path.join(data_dir, dataset, f'item_LF_{1-eval_ratio-test_ratio}.npz'))
    return item_sim
####################