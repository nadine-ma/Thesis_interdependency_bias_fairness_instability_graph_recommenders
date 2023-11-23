import collections
import math
import os
from argparse import ArgumentParser
from collections import Counter
import statistics
import numpy as np
from matplotlib import pyplot as plt


def plot_age_distribution(dataset):
    path_to_users_features = os.path.join("datasets", dataset, "user_features.csv")
    ages = []

    with open(path_to_users_features) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            age = parts[1]
            ages.append(age)

    counter_ages = Counter(ages)
    dict_ages = dict(counter_ages)
    dict_ages = collections.OrderedDict(sorted(dict_ages.items()))

    ages = list(dict_ages.keys())
    counts = list(dict_ages.values())
    fig = plt.figure(figsize=(3, 3))
    plt.bar(ages, counts, color ='maroon',
            width=0.8)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    plt.xlabel("Age categories", fontsize=10)
    plt.ylabel("User count", fontsize=10)
    plt.title(f"No. of {dataset} dataset users per age category", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_rating_distribution(dataset):
    path_to_ratings= os.path.join("datasets", dataset, "ratings_train_0.8.csv")
    ratings = []

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            ratings.append(int(parts[2]))

    counter_ratings = Counter(ratings)
    dict_ratings = dict(counter_ratings)
    dict_ratings = collections.OrderedDict(sorted(dict_ratings.items()))

    ratings = list(str(key) for key in dict_ratings.keys())
    counts = list(dict_ratings.values())
    fig = plt.figure(figsize=(5, 4))
    plt.bar(ratings, counts, color='maroon',
            width=0.7)

    plt.xlabel("Rating categories")
    plt.ylabel("No. of ratings")
    plt.title(f"No. of ratings per rating category in the {dataset} dataset")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution_per_age(dataset):
    path_to_users_features = os.path.join("datasets", dataset, "user_features.csv")
    ages = []

    with open(path_to_users_features) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            age = int(parts[1])
            ages.append(age)

    path_to_ratings = r"C:\Users\Nadine\Desktop\GitHub Repositories\Bias_Mitigation_Thesis\datasets\ml100k\ratings_train_0.8.csv"
    ratings_ages = {}

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            u_id = int(parts[0])
            try:
                age = int(ages[u_id])
            except:
                print()
            if age not in ratings_ages:
                ratings_ages[age] = []
            ratings_ages[age].append(parts[2])
    ratings_ages = collections.OrderedDict(sorted(ratings_ages.items()))

    counters = []
    for age in ratings_ages:
        dict_ratings = dict(Counter(ratings_ages[age]))
        dict_ratings = collections.OrderedDict(sorted(dict_ratings.items()))
        counters.append(dict_ratings)



    ratings = [1,2,3,4,5]
    X_axis = np.arange(len(ratings))

    ax = plt.subplot(111)
    ax.bar(X_axis, list(counters[0].values()), 0.1, label='<18')
    ax.bar(X_axis + 0.1, list(counters[1].values()), 0.1, label='18-24')
    ax.bar(X_axis + 0.2, list(counters[2].values()), 0.1, label='25-34')
    ax.bar(X_axis + 0.3, list(counters[3].values()), 0.1, label='35-44')
    ax.bar(X_axis + 0.4, list(counters[4].values()), 0.1, label='45-49')
    ax.bar(X_axis + 0.5, list(counters[5].values()), 0.1, label='50-55')
    ax.bar(X_axis + 0.6, list(counters[6].values()), 0.1, label='>56')

    ax.set_xticks(X_axis, ratings)
    ax.set_xlabel("Rating categories")
    ax.set_ylabel("Number of user ratings")
    ax.set_title(f"Number of user ratings per rating category in {dataset} dataset")
    ax.legend()
    plt.show()


def get_average_rating(dataset):
    import statistics
    path_to_ratings = os.path.join("datasets", dataset, "ratings_train_0.8.csv")
    ratings = []

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            ratings.append(int(parts[2]))

    av_rating = sum(ratings)/len(ratings)
    print(f"average rating: {av_rating}")



def average_ratings_per_age(dataset):
    path_to_ratings = os.path.join("datasets", dataset, "ratings_train_0.8.csv")
    path_to_features = os.path.join("datasets", dataset, "user_features.csv")

    users = []
    with open(path_to_features) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            age = parts[1]
            users.append(age)


    ratings = {}
    ages = ["1","18","25","35","45","50","56"]
    for age in ages:
        if age not in ratings:
            ratings[age] = []

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            user = int(parts[0])
            item = parts[1]
            rating = int(parts[2])
            age = users[user]
            ratings[age].append(rating)

    ratings_av = {}
    for age in ages:
        if age not in ratings_av:
            ratings_av[age] = 0

    for age in ratings:
        ratings_av[age] = np.mean(ratings[age])

    #av_rating = sum(ratings)/len(ratings)
    print(f"average ratings per age group: {ratings_av}")


def plot_longtail_distribution(dataset):

    path_to_ratings = os.path.join("datasets", dataset, "ratings_train_0.8.csv")

    users = []

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            users.append(int(parts[1]))

    counter_u = Counter(users)
    dict_u = dict(counter_u.most_common())

    ratings = np.arange(len(dict_u))
    counts = list(dict_u.values())
    fig = plt.figure(figsize=(4, 3))
    markerline, stemline, baseline, = plt.stem(ratings, counts,'maroon',basefmt=" ")
    plt.setp(markerline, markersize=1)


    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Items", fontsize=12)
    plt.ylabel("Rating count", fontsize=12)
    plt.title(f"Longtail distribution of the {dataset} dataset", fontsize=10)
    plt.tight_layout()
    plt.show()



def plot_ratings_per_user(dataset):
    path_to_ratings = os.path.join("datasets", dataset, "ratings_train_0.8.csv")

    users = []

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            users.append(int(parts[0]))

    counter_u = Counter(users)
    dict_u = dict(counter_u.most_common())

    ratings = np.arange(len(dict_u))
    counts = list(dict_u.values())
    fig = plt.figure(figsize=(6, 4))
    plt.bar(ratings, counts, color='maroon',
            width=1)

    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel("Users", fontsize=10)
    plt.ylabel("No. of ratings", fontsize=10)
    plt.title(f"No. of ratings per user in {dataset} dataset", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_correlation_rating_count_average_rating(dataset):

    path_to_ratings = os.path.join("datasets", dataset, "ratings_train_0.8.csv")

    items = {}

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            item = int(parts[1])
            rating = int(parts[2])
            if item not in items:
                items[item] = []
            items[item].append(rating)

    counts_i = []
    averages_i = []
    for item in items:
        counts_i.append(len(items[item]))
        averages_i.append(np.mean(items[item]))


    fig = plt.figure(figsize=(6, 6))
    plt.scatter(counts_i, averages_i, color="maroon")

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("rating count", fontsize=12)
    plt.ylabel("average rating", fontsize=12)
    plt.title(f"Correlation between popularity and average rating score of items", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_distribution_of_item_popularity_by_age_group(dataset):

    path_to_ratings = os.path.join("datasets", dataset, "ratings_train_0.8.csv")
    path_to_features = os.path.join("datasets", dataset, "user_features.csv")

    user_ages = {}
    with open(path_to_features) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            user_id = int(parts[0])-1
            age = int(parts[1])
            user_ages[user_id] = age

    items = {}
    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            item = int(parts[1])
            if item == 1655:
                print()
            rating = int(parts[2])
            if item not in items:
                items[item] = []
            items[item].append(rating)

    counts_i = {}
    for item in items:
        counts_i[item] = (len(items[item]))

    popularities_user_groups = {}

    with open(path_to_ratings) as f:
        for line in f:
            parts = line.strip('\n').split(',')
            item_id = int(parts[1])
            user_id = int(parts[0])
            if user_ages[user_id] not in popularities_user_groups:
                popularities_user_groups[user_ages[user_id]] = []
            try:
                popularities_user_groups[user_ages[user_id]].append(counts_i[item_id])
            except:
                print()

    for group in popularities_user_groups:
        plt.boxplot(popularities_user_groups[group], patch_artist=True)
        plt.title(f"Distribution of item popularity rated by age group {group}", fontsize=10)
        plt.show()





if __name__ == '__main__':
    function_map = {
        'plot_age_distribution': plot_age_distribution,
        'plot_rating_distribution': plot_rating_distribution,
        'plot_rating_distribution_per_age': plot_rating_distribution_per_age,
        'get_average_rating': get_average_rating,
        'get_average_ratings_per_age': average_ratings_per_age,
        'plot_ratings_per_user': plot_ratings_per_user,
        'plot_longtail_distribution': plot_longtail_distribution,
        'plot_correlation_ratings': plot_correlation_rating_count_average_rating,
        'boxplot_item_popularity_by_age_group': plot_distribution_of_item_popularity_by_age_group
    }

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--command", choices=function_map.keys())
    args = parser.parse_args()

    #args.command = 'plot_correlation_ratings1'

    func = function_map[args.command]
    func(args.dataset)
