import copy
import os

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings

from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

def calculate_similarity(variance_check_predictions: dict):
    cos_sims = []
    i = 1
    while i < len(variance_check_predictions):
        cos_sim = (np.dot(variance_check_predictions[0], variance_check_predictions[i]) /
                   (np.linalg.norm(variance_check_predictions[0]) * np.linalg.norm(variance_check_predictions[1])))
        cos_sims.append(round(cos_sim, 4))
        i += 1
    cos_sim = np.mean(cos_sims)
    print(
        f'Average cosine similarity variance over {len(variance_check_predictions)} evaluations = {cos_sims}')
    return cos_sims

def box_plot_variances(variance_check_predictions: list):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.boxplot(variance_check_predictions)
    ax.set_xlabel('model runs')
    ax.set_ylabel('rating predictions')
    plt.show()


def weighted_precision(labels, pre, classes=[]):
    return metrics.precision_score(labels, pre, average='weighted')

def weighted_recall(labels, pre, classes=[]):
    return metrics.recall_score(labels, pre, average='weighted')

def one_v_rest_auroc(labels, pre, classes=[]):
    scores = []
    for el in classes:
        fpr, tpr, thresholds = metrics.roc_curve(labels, pre, pos_label=el)
        scores.append(metrics.auc(fpr, tpr))
    scores = [item for item in scores if not (np.isnan(item)) == True]
    if len(scores) == 0:
        return 0
    else:
        return sum(scores) / len(scores)

def one_v_rest_weighted_auroc(labels, pre, classes=[]):
    scores = 0
    for el in classes:
        class_count = labels.count(el)
        fpr, tpr, thresholds = metrics.roc_curve(labels, pre, pos_label=el)
        scores = scores + (metrics.auc(fpr, tpr) * class_count)
    #scores = [item for item in scores if not (np.isnan(item)) == True]
    return scores / len(labels)

def one_v_rest_gender_auroc(labels, pre, classes=[]):
    #scores = []
    fpr_all = []
    tpr_all = []
    for gender in labels:
        #scores_gender = []
        fpr_gender = []
        tpr_gender = []
        for el in classes:
            fpr, tpr, thresholds = metrics.roc_curve(labels[gender], pre[gender], pos_label=el)
            #scores_gender.append(metrics.auc(fpr, tpr))
            fpr_gender.append(np.mean(fpr))
            tpr_gender.append(np.mean(tpr))
        #scores.append(sum(scores_gender) / len(scores_gender))
        fpr_gender = [item for item in fpr_gender if not (np.isnan(item)) == True]
        tpr_gender = [item for item in tpr_gender if not (np.isnan(item)) == True]
        fpr_all.append(sum(fpr_gender) / len(fpr_gender))
        tpr_all.append(sum(tpr_gender) / len(tpr_gender))
    tpr_gender = round(abs(1 - (tpr_all[0] / tpr_all[1])), 4)
    fpr_gender = round(abs(1 - (fpr_all[0] / fpr_all[1])), 4)
    return tpr_gender, fpr_gender

def one_v_rest_weighted_gender_auroc(labels, pre, classes=[]):
    #scores = []
    len_labels = 0
    fpr_all = []
    tpr_all = []
    class_count = {}
    for gender in labels:
        len_labels += len(labels[gender])
        for el in classes:
            if el not in class_count:
                class_count[el] = 0
            class_count[el] += (labels[gender].count(el))
    for gender in labels:
        #scores_gender = []
        fpr_gender = 0
        tpr_gender = 0
        for el in classes:
            fpr, tpr, thresholds = metrics.roc_curve(labels[gender], pre[gender], pos_label=el)
            #scores_gender.append(metrics.auc(fpr, tpr))
            fpr_gender = fpr_gender + (np.mean(fpr) * class_count[el])
            tpr_gender = tpr_gender + (np.mean(tpr) * class_count[el])
        fpr_all.append(fpr_gender / len_labels)
        tpr_all.append(tpr_gender / len_labels)
    tpr_gender = round(abs(1 - (tpr_all[0] / tpr_all[1])), 4)
    fpr_gender = round(abs(1 - (fpr_all[0] / fpr_all[1])), 4)
    return tpr_gender, fpr_gender


def get_popularity_distribution(uim_train):
    movies = {}
    for user in uim_train:
        rated_items = user.nonzero()[0]
        for movie_id in rated_items:
            if movie_id not in movies:
                movies[movie_id] = 0
            movies[movie_id] += 1
    movies = list(sorted(movies.items(), key=lambda item: item[0], reverse=False))

    ############## Plot popularity distribution / Long Tail
    #popularity_distribution = dict(movies)
    #import matplotlib.pyplot as plt
    #plt.plot(list(popularity_distribution.values()))
    #plt.show()
    ##############
    return movies


def calculate_longtail_percentage(popularity_distribution=list, topk_items=list):
    topk_short_head_items = 0
    short_head = copy.copy(popularity_distribution)
    count_short_head = int(round(len(short_head) * 0.2, 0))
    del short_head[count_short_head:len(short_head)]
    short_head = dict(short_head)
    ######
    #popularity_distribution = dict(popularity_distribution)
    #from numpy import trapz
    #print("whole area = " + str(trapz(np.array(list(popularity_distribution.values())))))
    #print("short head area = " + str(trapz(np.array(list(short_head.values())))))
    ######
    for item in topk_items:
        if item in short_head:
            topk_short_head_items += 1
    return (len(topk_items)-topk_short_head_items)/len(topk_items)

def rmse(predictions, targets):             #ToDo: remove?
    return np.sqrt(((predictions - targets) ** 2).mean())

def value_unfairness():
    pass


#https://github.com/Jenniferz28/Collaborative-Filtering-Recommendation/blob/master/ndcg.py
def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]    #rating
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def average_recommendation_popularity_single_user(item_popularity_distribution, user_top_k):
    sum = 0
    for item in user_top_k:
        try:
            sum += item_popularity_distribution[item][1]
        except:
            # If element has not been rated in training set, the popularity is assigned 0
            sum += 0

    return sum/len(user_top_k)




