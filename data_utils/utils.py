import copy
import csv
import os
import random
from collections import defaultdict
from operator import itemgetter



def get_total_user_item_counts(data_dir, dataset):

    with open(os.path.join(data_dir, dataset, 'users_remapped.csv')) as f:
        user_count = sum(1 for row in f)
    with open(os.path.join(data_dir, dataset, 'items_remapped.csv')) as f:
        movie_count = sum(1 for row in f)
    return user_count, movie_count

def generate_user_item_dicts(data_dir, dataset, eval_ratio, test_ratio, train=True, eval=False, test=False):  # FairGo
    user_set = defaultdict(set)
    item_set = defaultdict(set)
    len_data = 0
    path = ""
    if test:
        path = "test"
    elif eval:
        path = f"eval_{eval_ratio}"
    elif train:
        path = f"train_{1-eval_ratio-test_ratio}"

    with open(os.path.join(data_dir, dataset, 'ratings_' + path + '.csv')) as f:
        for idx, line in enumerate(f):
            parts = line.strip('\n').split(',')
            user = int(parts[0])
            movie_id = int(parts[1])
            user_set[user].add(movie_id)
            item_set[movie_id].add(user)
            len_data += 1
    return user_set, item_set, len_data

def combine_user_rating_sets(train_set, eval_set, test_set):
    user_set_all = copy.deepcopy(train_set)
    if eval_set is not None:
        for k, v in eval_set.items():
            if k in user_set_all:
                user_set_all[k].update(eval_set[k])
            else:
                user_set_all[k] = set()
                user_set_all[k].update(eval_set[k])
    if test_set is not None:
        for k, v in test_set.items():
            if k in user_set_all:
                user_set_all[k].update(test_set[k])
            else:
                user_set_all[k] = set()
                user_set_all[k].update(test_set[k])
    return user_set_all


def generate_rating_list(uim):
    ratings = []
    for idx, user in enumerate(uim):
        rated_items = user.nonzero()[0]
        for movie_id in rated_items:
            rating = user[movie_id]
            ratings.append([idx, movie_id, rating])

    return ratings


def generate_rating_list_oversampled(data_dir, dataset, eval_ratio, test_ratio, oversampling_feature):
    ratings = []
    with open(os.path.join(data_dir, dataset, f'ratings_train_{1-eval_ratio-test_ratio}_oversampled_{oversampling_feature}.csv')) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            user_id = int(parts[0])
            item_id = int(parts[1])
            rating = int(parts[2])
            ratings.append([user_id, item_id, rating])
    return ratings

def generate_user_list(data_dir, dataset, train=True, eval=False):
    users = []
    path = 'train'
    if eval:
        path = 'eval'
    with open(os.path.join(data_dir, dataset, 'users_fair_' + path + '.csv')) as f:
        for line in f:
            users.append(int(line.strip('\n')))
    return users


def sample_random_items(data_dir, dataset, item_num, no_random_items):
    value_unfairness_items = []
    while len(value_unfairness_items) < no_random_items:
        random_item = random.randrange(item_num)
        while random_item in value_unfairness_items:
            random_item = random.randrange(item_num)
        value_unfairness_items.append(random_item)

    with open(os.path.join(data_dir, dataset, f'random_items_{no_random_items}.txt'), 'w', newline='') as f:
        f.write(','.join([str(x) for x in value_unfairness_items]))
    return value_unfairness_items


def load_random_items(data_dir, dataset, item_num, no_random_items):
    if not os.path.exists(os.path.join(data_dir, dataset, f'random_items_{no_random_items}.txt')):
        return sample_random_items(data_dir, dataset, item_num, no_random_items)
    else:
        with open(os.path.join(data_dir, dataset, f'random_items_{no_random_items}.txt')) as f:
            items = f.read()
            items = items.strip('\n').split(',')
            items = [int(item) for item in items]
        return items


def write_eval_measures(log_dir, values_dict, test=False):
    file_name = 'eval_measures.csv'
    if test:
        file_name = 'test_measures.csv'
    with open(os.path.join(log_dir, file_name), 'a',  newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(values_dict)

