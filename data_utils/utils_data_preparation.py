import csv
import os
import random
from operator import itemgetter
import numpy as np
import re

import scipy
from tqdm import tqdm

def id_remapping_users(data_dir, dataset, user_train_eval_split=True):

    if not os.path.exists(os.path.join(data_dir, dataset, 'users_remapped.csv')):
        print("id remapping users")

        # *1: "Under 18"
        # *18: "18-24"
        # *25: "25-34"
        # *35: "35-44"
        # *45: "45-49"
        # *50: "50-55"
        # *56: "56+"

        age_ranges = {}
        for el in range(0, 150):
            if 0 <= el < 18:
                age_ranges[el] = 1
            elif el >= 18 and el < 25:
                age_ranges[el] = 18
            elif el >= 25 and el < 35:
                age_ranges[el] = 25
            elif el >= 35 and el < 45:
                age_ranges[el] = 35
            elif el >= 45 and el < 50:
                age_ranges[el] = 45
            elif el >= 50 and el < 56:
                age_ranges[el] = 50
            elif el >= 56:
                age_ranges[el] = 56

        user_features = []
        ages = []
        users = []
        filename = ""
        separator = ""
        user_id_pos = 0
        age_pos = None
        encoding = "ISO-8859-1"

        if dataset == "ml100k":
            filename = 'u.user'
            separator = "|"
            user_id_pos = 0
            age_pos = 1

        elif dataset == "ml1m":
            filename = 'users.dat'
            separator = "::"
            user_id_pos = 0
            age_pos = 2
            encoding = "ISO-8859-1"

        elif dataset == "bookCrossing":
            filename = "BX-Users.csv"
            separator = ";"
            user_id_pos = 0
            age_pos = 2

            with open(os.path.join(data_dir, dataset, "BX-Book-Ratings_clean.csv")) as f:
                for line in tqdm(f):
                    parts = line.strip('\n').split(';')
                    u_id = int(parts[0])
                    if u_id not in users:
                        users.append(u_id)

            ######################################

        with open(os.path.join(data_dir, dataset, filename), encoding=encoding) as f:
            for line in f:
                parts = line.strip('\n').split(separator)
                if dataset == "bookCrossing":
                    user_id = int(parts[user_id_pos].strip('"'))
                    if user_id in users:
                        try:
                            age = int(parts[age_pos].strip('"'))
                        except:
                            age = -1
                        age_id = 0
                        if age in age_ranges:
                            age_id = age_ranges[age]
                        user_features.append([user_id, age_id])

                else:
                    user_id = int(parts[user_id_pos])
                    age = int(parts[age_pos])
                    if user_id not in users:
                        users.append(user_id)

                    age_id = 0
                    if age in age_ranges:
                        age_id = age_ranges[age]
                    user_features.append([user_id, age_id])




        with open(os.path.join(data_dir, dataset, 'user_features.csv'), 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for user in user_features:
                csvwriter.writerow(user)

        with open(os.path.join(data_dir, dataset, 'users_remapped.csv'), 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for idx, user in enumerate(user_features):
                csvwriter.writerow([idx, user[0]])

        print("finished user remapping")

    if user_train_eval_split:
        if not os.path.exists(os.path.join(data_dir, dataset, 'users_fair_train.csv')):
            eval_users = random.sample([u for u in range(len(users))], round(len(users) * 0.2))
            train_users = [el for el in range(len(users)) if el not in eval_users]
            with open(os.path.join(data_dir, dataset, 'users_fair_eval.csv'), 'w', newline='') as f:
                csvwriter = csv.writer(f)
                for user in eval_users:
                    csvwriter.writerow([user])

            with open(os.path.join(data_dir, dataset, 'users_fair_train.csv'), 'w', newline='') as f:
                csvwriter = csv.writer(f)
                for user in train_users:
                    csvwriter.writerow([user])
            print("finished user train test split")




def id_remapping_items(data_dir, dataset):
    if not os.path.exists(os.path.join(data_dir, dataset, 'items_remapped.csv')):
        print("id remapping items")
        items = []
        movie_features = []
        item_id_pos_ratings = 1
        encoding = "ISO-8859-1"

        if dataset == "ml100k":
            filename_items = 'u.item'
            filename_ratings = 'u.data'
            separator_items = "|"
            separator_ratings = '\t'
            item_id_pos = 0

        elif dataset == "ml1m":
            filename_items = 'movies.dat'
            filename_ratings = 'ratings.dat'
            separator_items = "::"
            separator_ratings = '::'
            item_id_pos = 1

        elif dataset == "bookCrossing":
            filename_items = "BX-Books.csv"
            filename_ratings = "BX-Book-Ratings_clean.csv"
            separator_items = ";"
            separator_ratings = ';'
            item_id_pos = 1

        with open(os.path.join(data_dir, dataset, filename_ratings),encoding=encoding) as f:
            for line in tqdm(f):
                parts = line.strip('\n').split(separator_ratings)
                item_identifier = parts[item_id_pos_ratings]
                if item_identifier not in items:
                    items.append(item_identifier)

        with open(os.path.join(data_dir, dataset, filename_items), encoding=encoding) as f:
            for line in tqdm(f):
                parts = line.strip('\n').split(separator_items)
                item_identifier = parts[item_id_pos]
                if dataset == "bookcrossing":
                    item_identifier = item_identifier.strip('"')
                if item_identifier in items:
                    movie = [items.index(item_identifier)]
                    movie_features.append(movie)


        with open(os.path.join(data_dir, dataset, 'items_remapped.csv'), 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for idx, item in enumerate(items):
                csvwriter.writerow([idx, item])
        print("finished item remapping")



def remove_implicit_ratings_BookCrossing(data_dir, dataset, min_ratings_per_user):
    if not os.path.exists(os.path.join(data_dir, dataset, f'BX-Book-Ratings_clean.csv')):
        print(f"removing implicit ratings and ratings by users with less than {min_ratings_per_user} ratings")
        ratings = []
        ratings_clean = []
        counter_users = {}

        with open(os.path.join(data_dir, dataset, "BX-Book-Ratings.csv"), encoding="ISO-8859-1") as f:
            for line in tqdm(f):
                parts = line.strip('\n').split(';')
                user = parts[0].strip('"')
                rating = int(parts[2].strip('"'))
                isbn = re.sub('[^A-Za-z0-9]+', '', parts[1])
                if rating > 0:
                    ratings.append([user,isbn,rating])
                    if user not in counter_users:
                        counter_users[user] = 0
                    counter_users[user] += 1

        counter_users_sorted = sorted(counter_users.items(), key=lambda x: x[1])
        users_more_than = [i[0] for i in counter_users_sorted if i[1] >=min_ratings_per_user]

        for el in tqdm(ratings):
            if el[0] in users_more_than:
                ratings_clean.append(el)

        with open(os.path.join(data_dir, dataset, f'BX-Book-Ratings_clean.csv'), 'w', newline='') as f:
            for el in ratings_clean:
                user = el[0]
                isbn = el[1]
                rating = el[2]
                f.write(f'{user};{isbn};{rating}\n')



def split_rating_data(data_dir, dataset, eval_ratio=0.1, test_ratio=0.1):
    if not os.path.exists(os.path.join(data_dir, dataset, f'ratings_train_{1-eval_ratio-test_ratio}.csv')): #f'ratings_train_{1-eval_ratio-test_ratio}.csv')):
        print("split rating data")

        if dataset == "ml100k":
            filename = 'u.data'
            separator = '\t'
        elif dataset == "ml1m":
            filename = 'ratings.dat'
            separator = '::'
        elif dataset == "bookCrossing":
            filename = 'BX-Book-Ratings_clean.csv'
            separator = ';'


        user_mapping = {}
        with open(os.path.join(data_dir, dataset, 'users_remapped.csv')) as f:
            for line in f:
                parts = line.strip("\n").split(',')
                user_id_remapped = parts[0]
                user_id = parts[1]
                user_mapping[user_id] = user_id_remapped

        item_mapping = {}
        with open(os.path.join(data_dir, dataset, 'items_remapped.csv')) as f:
            for line in f:
                parts = line.strip("\n").split(',')
                item_identifier_remapped = parts[0]
                item_identifier = parts[1]
                item_mapping[item_identifier] = item_identifier_remapped

        ratings = []

        if dataset == "bookCrossing":
            with open(os.path.join(data_dir, dataset, "BX-Book-Ratings_clean.csv")) as f:
                for idx, line in enumerate(f):
                    parts = line.strip('\n').split(separator)
                    user = parts[0]
                    user = user_mapping[user]
                    item_identifier = parts[1]
                    item_identifier = item_mapping[item_identifier]
                    rating = int(parts[2])
                    if rating > 0:
                        ratings.append([user, item_identifier, rating])
        else:
            with open(os.path.join(data_dir, dataset, filename)) as f:
                for line in f:
                    parts = line.strip('\n').split(separator)
                    user = parts[0]
                    try:
                        user = user_mapping[user]
                    except:
                        print()
                    item_id = parts[1]
                    item_id = item_mapping[item_id]
                    rating = parts[2]
                    ratings.append([user, item_id, rating])

        ratings = np.array(ratings)

        with open(os.path.join(data_dir, dataset, 'ratings_remapped.csv'), 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for item in ratings:
                csvwriter.writerow(item)

        eval_idx = np.random.choice(ratings.shape[0], size=int(ratings.shape[0] * eval_ratio), replace=False)
        left = set(range(ratings.shape[0])) - set(eval_idx)
        test_idx = np.random.choice(list(left), size=int(ratings.shape[0] * test_ratio), replace=False)
        train_idx = list(left - set(test_idx))

        with open(os.path.join(data_dir, dataset, f'ratings_train_{1-eval_ratio-test_ratio}.csv'), 'w', newline='') as f: #f'ratings_train_{1-eval_ratio-test_ratio}.csv'), 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for i in train_idx:
                csvwriter.writerow(ratings[i])

        with open(os.path.join(data_dir, dataset, f'ratings_eval_{eval_ratio}.csv'), 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for i in eval_idx:
                csvwriter.writerow(ratings[i])

        if test_ratio > 0.0:
            with open(os.path.join(data_dir, dataset, f'ratings_test_{test_ratio}.csv'), 'w', newline='') as f:
                csvwriter = csv.writer(f)
                for i in test_idx:
                    csvwriter.writerow(ratings[i])
        print("finished data splitting")


def load_user_features(data_dir, dataset):
    user_features = []
    filename = 'user_features.csv'
    with open(os.path.join(data_dir, dataset, filename)) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            age = int(parts[1])
            user_features.append(age)
    return np.array(user_features)

