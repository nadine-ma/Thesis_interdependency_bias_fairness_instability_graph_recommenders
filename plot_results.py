import ast
import itertools
import os
from argparse import ArgumentParser

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def get_data(dataset, opt_measures, type="instances"):
    all = {}

    for model in ['DiversityTradeoff', 'FlexibleFairness']:
        if model not in all:
            all[model] = {}
        for opt_measure in opt_measures[model]:
            run_dirs = []
            evaluation_measures = []
            try:
                with open(os.path.join("logs", dataset, model, f'{type}_{opt_measure}_tested.txt')) as f:
                    for line in f:
                        vals = {}
                        parts = line.strip("\n").strip('"').split(",")
                        for part in parts:
                            name = part.split(":")[0]
                            value = part.split(":")[1]
                            if name == "run_dir":
                                run_dirs.append(value)
                            elif name == "epoch":
                                pass
                            else:
                                vals[name] = value
                        evaluation_measures.append(vals)

                all[model][opt_measure] = [run_dirs, evaluation_measures]
            except:
                pass
    return all


def create_test_measures_dict(dataset, opt_measures, type="instances"):
    all = get_data(dataset, opt_measures, type=type)

    alphabets = ['RMSE', 'AD@20', 'ID@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']

    label_dict = {'RMSE': 'RMSE',
                  'AD@20': 'AD@20',
                  'ID@20': 'ID@20',
                  'ARP@20': 'ARP@20',
                  'RMSE_coeff_var': '$CV_{RMSE}$',
                  'VU_coeff_var': '$CV_{VU}$'
                  }

    test_measures_dict = {}
    test_measures_counts_per_opt = []

    vus_all = {}
    rmses_all = {}
    for model in all:
        for optimised_val in all[model]:
            run_dirs = all[model][optimised_val][0]
            count_per_opt = 0
            for idx, run_dir in enumerate(run_dirs):
                count_per_opt += 1
                rmses = []
                vus = []
                for el in all[model][optimised_val][1][idx]:
                    val_name = el
                    val_value = all[model][optimised_val][1][idx][el]
                    if val_name not in test_measures_dict:
                        test_measures_dict[val_name] = []
                    if "RMSE_" in val_name and val_name != "RMSE_coeff_var":
                        rmses.append(float(val_value))
                        if val_name.split("_")[1] not in rmses_all:
                            rmses_all[val_name.split("_")[1]] = []
                        rmses_all[val_name.split("_")[1]].append(float(val_value))
                    elif "VU_" in val_name and val_name != "VU_coeff_var":
                        vus.append(float(val_value))
                        if val_name.split("_")[1] not in vus_all:
                            vus_all[val_name.split("_")[1]] = []
                        vus_all[val_name.split("_")[1]].append(float(val_value))
                    elif val_name == "ID@20":
                        if dataset == "bookCrossing":
                            pass
                        else:
                            test_measures_dict[val_name].append(float(val_value))
                    elif val_name == "F1_age":
                        pass
                    elif val_name == "best_test_loss":
                        pass
                    elif val_name == "best_test_loss_epoch":
                        pass
                    elif val_name == "max_F1_age":
                        pass
                    else:
                        test_measures_dict[val_name].append(float(val_value))
            test_measures_counts_per_opt.append(count_per_opt)

    return test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict


def plot_correlation_matrix(dataset, opt_measures):
    test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                        opt_measures)

    for idx, i in enumerate(["DiversityTradeoff", "FlexibleFairness", "both models"]):
        leave_loop = False
        if dataset == "bookCrossing":
            if idx == 0:
                i = "FlexibleFairness"
                start = 0
                end = sum(test_measures_counts_per_opt[:2])
                leave_loop = True

        else:
            if idx == 0:
                start = 0
                end = sum(test_measures_counts_per_opt[:3])
            elif idx == 1:
                start = sum(test_measures_counts_per_opt[:3])
                end = sum(test_measures_counts_per_opt[:5])
            elif idx == 2:
                start = 0
                end = sum(test_measures_counts_per_opt[:5])

        if dataset == "bookCrossing":
            alphabets = ['RMSE', 'AD@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
            corrs = np.zeros((len(alphabets), len(alphabets)))
            for idx, el in enumerate(alphabets):
                corrs[idx][idx] = 1
        else:
            corrs = np.zeros((len(alphabets), len(alphabets)))
            for idx, el in enumerate(alphabets):
                corrs[idx][idx] = 1

        combinations = itertools.combinations(alphabets, 2)
        for combi in combinations:
            res = spearmanr(np.array(test_measures_dict[combi[0]][start:end]),
                            np.array(test_measures_dict[combi[1]][start:end]))
            corr = res.statistic
            corrs[alphabets.index(combi[0])][alphabets.index(combi[1])] = corr
            corrs[alphabets.index(combi[1])][alphabets.index(combi[0])] = corr
            p = res.pvalue
            print(f"{combi} corr = {corr}, p = {p}")
        print()

        fig = plt.figure()
        fig_all = plt.imshow(corrs, cmap='RdBu',
                             vmin=-1,
                             vmax=1)

        fig.color_bar = fig.colorbar(fig_all,
                                     extend='both')

        for t in fig.color_bar.ax.get_yticklabels():
            t.set_fontsize(9)

        plt.title(f"{i} {dataset}")
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.xticks(range(len(alphabets)), [label_dict[i] for i in alphabets])
        plt.yticks(range(len(alphabets)), [label_dict[i] for i in alphabets])
        plt.show()
        if leave_loop:
            break


def scatter_plot(dataset, opt_measures):
    test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                        opt_measures,
                                                                                                        type="instances")
    if dataset == "bookCrossing":
        alphabets = ['RMSE', 'AD@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
    combinations = itertools.combinations(alphabets, 2)

    for combi in combinations:
        if dataset != "bookCrossing":
            start = 0
            end = sum(test_measures_counts_per_opt[:1])
            fig = plt.figure(figsize=(3.3, 3), constrained_layout=True)
            x_RMSE = np.array(test_measures_dict[combi[0]][start:end])
            y_RMSE = np.array(test_measures_dict[combi[1]][start:end])
            plt_RMSE = plt.scatter(x_RMSE, y_RMSE, color="red")
            plt_RMSE.set_color("#6b0362")

            start = sum(test_measures_counts_per_opt[:1])
            end = sum(test_measures_counts_per_opt[:2])
            x_AD = np.array(test_measures_dict[combi[0]][start:end])
            y_AD = np.array(test_measures_dict[combi[1]][start:end])
            plt_AD = plt.scatter(x_AD, y_AD, color="blue")
            plt_AD.set_color("#ff00ea")

            start = sum(test_measures_counts_per_opt[:2])
            end = sum(test_measures_counts_per_opt[:3])
            x_ID = np.array(test_measures_dict[combi[0]][start:end])
            y_ID = np.array(test_measures_dict[combi[1]][start:end])
            plt_ID = plt.scatter(x_ID, y_ID, color="green")
            plt_ID.set_color("#c476be")

        if dataset == "bookCrossing":
            start = 0
            end = sum(test_measures_counts_per_opt[:1])
        else:
            start = sum(test_measures_counts_per_opt[:3])
            end = sum(test_measures_counts_per_opt[:4])
        x_RMSE_F = np.array(test_measures_dict[combi[0]][start:end])
        y_RMSE_F = np.array(test_measures_dict[combi[1]][start:end])
        plt_RMSE_F = plt.scatter(x_RMSE_F, y_RMSE_F, color="red")
        plt_RMSE_F.set_color("#285e2d")

        if dataset == "bookCrossing":
            start = sum(test_measures_counts_per_opt[:1])
            end = sum(test_measures_counts_per_opt[:2])
        else:
            start = sum(test_measures_counts_per_opt[:4])
            end = sum(test_measures_counts_per_opt[:5])
        x_F1 = np.array(test_measures_dict[combi[0]][start:end])
        y_F1 = np.array(test_measures_dict[combi[1]][start:end])
        plt_F1 = plt.scatter(x_F1, y_F1, color="blue")
        plt_F1.set_color("#46ab50")

        plt.yticks(fontsize=9)
        plt.legend((plt_RMSE, plt_AD, plt_ID, plt_RMSE_F, plt_F1),
                   ("RMSE optimised (Diversity Tradeoff Model)",
                    "AD@20 optimised (Diversity Tradeoff Model)",
                    "ID@20 optimised (Diversity Tradeoff Model)",
                    "RMSE optimised (Compositional Fairness Model)",
                    "F1 optimised (Compositional Fairness Model)"),
                   scatterpoints=1,
                   loc="best", ncol=2, fontsize=8,
                   # bbox_to_anchor=(0.5, -0.3)
                   )

        plt.title(dataset)
        plt.xlabel(label_dict[combi[0]], fontsize=10)
        plt.ylabel(label_dict[combi[1]], fontsize=10)
        plt.tight_layout()
        plt.show()


def plot_prediction_disagreement(dataset, opt_measures):
    all = get_data(dataset, opt_measures, type="instances")

    pred_disagr_list = [0, 0, 0, 0, 0]

    for model in all:
        for optimised_val in all[model]:
            run_dirs = all[model][optimised_val][0]
            with open(os.path.join("logs", dataset, model, f'prediction_disagreements.txt')) as f:
                for line in f:
                    parts = line.strip("\n").split(":")
                    opt_measure = parts[0]
                    value = float(parts[1])
                    if opt_measure == "AD@20":
                        pred_disagr_list[1] = value
                    elif opt_measure == "ID@20":
                        pred_disagr_list[2] = value
                    elif opt_measure == "F1_age":
                        pred_disagr_list[4] = value
                    elif opt_measure == "RMSE":
                        if model == "DiversityTradeoff":
                            pred_disagr_list[0] = value
                        else:
                            pred_disagr_list[3] = value

    fig = plt.figure(figsize=(3, 3), constrained_layout=True)
    barlist = plt.bar(["RMSE_T", "AD_T", "ID_T", "RMSE_F", "F1_F"], pred_disagr_list)
    barlist[0].set_color("#6b0362")
    barlist[1].set_color("#ff00ea")
    barlist[2].set_color("#c476be")
    barlist[3].set_color("#285e2d")
    barlist[4].set_color("#46ab50")
    for el in barlist:
        el.set_edgecolor("black")
    plt.xticks([])
    plt.ylabel("Prediction Disagreement", fontsize=10)
    plt.yticks(fontsize=9)

    plt.legend((barlist[0], barlist[1], barlist[2], barlist[3], barlist[4]),
               ("RMSE optimised (Diversity Tradeoff Model)",
                "AD@20 optimised (Diversity Tradeoff Model)",
                "ID@20 optimised (Diversity Tradeoff Model)",
                "RMSE optimised (Compositional Fairness Model)",
                "F1 optimised (Compositional Fairness Model)"),
               loc="best", ncol=2, fontsize=8, )
    # bbox_to_anchor=(0.5, -0.27))
    plt.title(dataset)
    plt.tight_layout()

    plt.show()


def plot_averages(dataset, opt_measures):
    alphabets = ['RMSE', 'AD@20', 'ID@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
    width = 0.8
    if dataset != "bookCrossing":
        ind = np.arange(5)

        for el in alphabets:
            fig = plt.figure()
            test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                                opt_measures,
                                                                                                                type="instances")
            averages = {}
            if el not in averages:
                averages[el] = []

            averages[el].append(np.mean(
                test_measures_dict[el][0:sum(test_measures_counts_per_opt[:1])]))
            averages[el].append(np.mean(
                test_measures_dict[el][
                sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])]))
            averages[el].append(np.mean(
                test_measures_dict[el][
                sum(test_measures_counts_per_opt[:2]):sum(test_measures_counts_per_opt[:3])]))
            averages[el].append(np.mean(
                test_measures_dict[el][
                sum(test_measures_counts_per_opt[:3]):sum(test_measures_counts_per_opt[:4])]))
            averages[el].append(np.mean(
                test_measures_dict[el][
                sum(test_measures_counts_per_opt[:4]):sum(test_measures_counts_per_opt[:5])]))

            barlist = plt.bar(ind, averages[el], width=width)

            plt.xlabel(dataset, fontsize=10)

            barlist[0].set_color("#6b0362")
            barlist[0].set_edgecolor("black")
            barlist[1].set_color("#ff00ea")
            barlist[1].set_edgecolor("black")
            barlist[2].set_color("#c476be")
            barlist[2].set_edgecolor("black")
            barlist[3].set_color("#285e2d")
            barlist[3].set_edgecolor("black")
            barlist[4].set_color("#46ab50")
            barlist[4].set_edgecolor("black")

            plt.xticks([])
            plt.yticks(fontsize=9)
            plt.ylabel(label_dict[el], fontsize=10)

            plt.legend((barlist[0], barlist[1], barlist[2], barlist[3], barlist[4]),
                       ("RMSE optimised (Diversity Tradeoff Model)",
                        "AD@20 optimised (Diversity Tradeoff Model)",
                        "ID@20 optimised (Diversity Tradeoff Model)",
                        "RMSE optimised (Compositional Fairness Model)",
                        "F1 optimised (Compositional Fairness Model)",
                        ),
                       loc="best", ncol=3, fontsize=10,
                       # bbox_to_anchor=(0.0, -0.27),
                       facecolor="white")
            plt.tight_layout()
            plt.show()

    if dataset == "bookCrossing":
        alphabets = ['RMSE', 'AD@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
        ind = np.arange(2)
        for el in alphabets:
            fig = plt.figure()

            test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                                opt_measures,
                                                                                                                type="instances")
            averages = {}
            # for el in alphabets:
            if el not in averages:
                averages[el] = []

            averages[el].append(np.mean(
                test_measures_dict[el][0:sum(test_measures_counts_per_opt[:1])]))
            averages[el].append(np.mean(
                test_measures_dict[el][
                sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])]))

            barlist = plt.bar(ind, averages[el], width=width)
            plt.xlabel(dataset, fontsize=10)
            barlist[0].set_color("#285e2d")
            barlist[0].set_edgecolor("black")
            barlist[1].set_color("#46ab50")
            barlist[1].set_edgecolor("black")

            plt.xticks([])
            plt.yticks(fontsize=9)
            plt.ylabel(label_dict[el], fontsize=10)

            plt.legend((barlist[0], barlist[1]),
                       ("RMSE optimised (Compositional Fairness Model)",
                        "F1 optimised (Compositional Fairness Model)",
                        ),
                       loc="best", ncol=2, fontsize=10,
                       # bbox_to_anchor=(0.0, -0.27),
                       facecolor="white")
            plt.tight_layout()
            plt.show()


def plot_boxplot(dataset, opt_measures):
    alphabets = ['RMSE', 'AD@20', 'ID@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
    width = 0.7
    if dataset != "bookCrossing":

        for el in alphabets:
            fig = plt.figure()
            test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                                opt_measures,
                                                                                                                type="instances")
            averages = {}
            if el not in averages:
                averages[el] = []

            if dataset != "bookCrossing":
                averages[el].append(
                    test_measures_dict[el][0:sum(test_measures_counts_per_opt[:1])])
                averages[el].append(
                    test_measures_dict[el][
                    sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])])
                averages[el].append(
                    test_measures_dict[el][
                    sum(test_measures_counts_per_opt[:2]):sum(test_measures_counts_per_opt[:3])])
                averages[el].append(
                    test_measures_dict[el][
                    sum(test_measures_counts_per_opt[:3]):sum(test_measures_counts_per_opt[:4])])
                averages[el].append(
                    test_measures_dict[el][
                    sum(test_measures_counts_per_opt[:4]):sum(test_measures_counts_per_opt[:5])])

                barlist = plt.boxplot(averages[el], patch_artist=True, widths=width)

                plt.xlabel(dataset, fontsize=10)

                colors = ["#6b0362", "#ff00ea", "#c476be", "#285e2d", "#46ab50"]
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(barlist[element], color="black")

                for idx, patch in enumerate(barlist["boxes"]):
                    patch.set_facecolor(colors[idx])

                plt.xticks([])
                plt.yticks(fontsize=9)
                plt.ylabel(label_dict[el], fontsize=10)

                plt.legend((barlist["boxes"]),
                           ("RMSE optimised (Diversity Tradeoff Model)",
                            "AD@20 optimised (Diversity Tradeoff Model)",
                            "ID@20 optimised (Diversity Tradeoff Model)",
                            "RMSE optimised (Compositional Fairness Model)",
                            "F1 optimised (Compositional Fairness Model)",
                            ),
                           loc="best", ncol=1, fontsize=8,
                           # bbox_to_anchor=(0.0, -0.27),
                           facecolor="white")
                # plt.tight_layout()
                plt.show()

    if dataset == "bookCrossing":
        alphabets = ['RMSE', 'AD@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
        for el in alphabets:
            fig = plt.figure()

            test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                                opt_measures,
                                                                                                                type="instances")
            averages = {}
            # for el in alphabets:
            if el not in averages:
                averages[el] = []

            averages[el].append(
                test_measures_dict[el][0:sum(test_measures_counts_per_opt[:1])])
            averages[el].append(
                test_measures_dict[el][
                sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])])

            barlist = plt.boxplot(averages[el], patch_artist=True, widths=width)
            plt.xlabel(dataset, fontsize=10)

            colors = ["#285e2d", "#46ab50"]
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(barlist[element], color="black")

            for idx, patch in enumerate(barlist["boxes"]):
                patch.set_facecolor(colors[idx])

            plt.xticks([])
            plt.yticks(fontsize=9)
            plt.ylabel(label_dict[el], fontsize=10)

            plt.legend((barlist["boxes"]),
                       ("RMSE optimised (Compositional Fairness Model)",
                        "F1 optimised (Compositional Fairness Model)",
                        ),
                       loc="best", ncol=1, fontsize=8,
                       # bbox_to_anchor=(0.0, -0.27),
                       facecolor="white")
            plt.tight_layout()
            plt.show()


################################################################################################################################

def plot_alpha_variants(dataset, opt_measures):
    alphabets = ['RMSE', 'AD@20', 'ID@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']

    for el in alphabets:
        fig = plt.figure()
        test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                            opt_measures,
                                                                                                            type="alphas")
        values = {}
        for alpha in test_measures_dict['alpha']:
            if alpha not in values:
                values[alpha] = []

        for idx, el1 in enumerate(test_measures_dict[el][0:sum(test_measures_counts_per_opt[:1])]):
            alpha = test_measures_dict['alpha'][0:sum(test_measures_counts_per_opt[:1])][idx]
            values[alpha].append(el1)
        for idx, el1 in enumerate(test_measures_dict[el][
                                  sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])]):
            alpha = test_measures_dict['alpha'][
                    sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])][idx]
            values[alpha].append(el1)
        for idx, el1 in enumerate(test_measures_dict[el][
                                  sum(test_measures_counts_per_opt[:2]):sum(test_measures_counts_per_opt[:3])]):
            alpha = test_measures_dict['alpha'][
                    sum(test_measures_counts_per_opt[:2]):sum(test_measures_counts_per_opt[:3])][idx]
            values[alpha].append(el1)

        import collections
        values = collections.OrderedDict(sorted(values.items()))

        dict_alphas = {
            0: [],
            1: [],
            2: []
        }
        alphas = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        for alpha in values:
            for idx, alpha_val in enumerate(values[alpha]):
                dict_alphas[idx].append(alpha_val)

        plt_RMSE = plt.plot(alphas, dict_alphas[0], marker='o', linestyle='-',
                            label="RMSE optimised (Diversity Tradeoff Model)", color="#6b0362")
        plt_AD = plt.plot(alphas, dict_alphas[1], marker='^', linestyle='-',
                          label="AD@20 optimised (Diversity Tradeoff Model)",
                          color="#ff00ea")
        plt_ID = plt.plot(alphas, dict_alphas[2], marker='x', linestyle='-',
                          label="ID@20 optimised (Diversity Tradeoff Model)",
                          color="#c476be")

        plt.ylabel(label_dict[el], fontsize=10)
        plt.yticks(fontsize=9)
        plt.xlabel("α", fontsize=10)
        plt.xticks(fontsize=9)
        plt.title(dataset)
        plt.legend(loc="best", ncol=1, fontsize=8)
        #
        # plt.tight_layout()
        plt.show()


def plot_gamma_variants(dataset, opt_measures):
    alphabets = ['RMSE', 'AD@20', 'ID@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
    if dataset == "bookCrossing":
        alphabets = ['RMSE', 'AD@20', 'ARP@20', 'RMSE_coeff_var', 'VU_coeff_var']
    for el in alphabets:
        fig = plt.figure()
        test_measures_dict, alphabets, test_measures_counts_per_opt, label_dict = create_test_measures_dict(dataset,
                                                                                                            opt_measures,
                                                                                                            type="gammas")
        values = {}
        for gamma in test_measures_dict['gamma']:
            if gamma not in values:
                values[gamma] = []

        for idx, el1 in enumerate(test_measures_dict[el][0:sum(test_measures_counts_per_opt[:1])]):
            gamma = test_measures_dict['gamma'][0:sum(test_measures_counts_per_opt[:1])][idx]
            values[gamma].append(el1)
        for idx, el1 in enumerate(test_measures_dict[el][
                                  sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])]):
            gamma = test_measures_dict['gamma'][
                    sum(test_measures_counts_per_opt[:1]):sum(test_measures_counts_per_opt[:2])][idx]
            values[gamma].append(el1)
        for idx, el1 in enumerate(test_measures_dict[el][
                                  sum(test_measures_counts_per_opt[:2]):sum(test_measures_counts_per_opt[:3])]):
            gamma = test_measures_dict['gamma'][
                    sum(test_measures_counts_per_opt[:2]):sum(test_measures_counts_per_opt[:3])][idx]
            values[gamma].append(el1)

        import collections
        values = collections.OrderedDict(sorted(values.items()))

        dict_gammas = {
            0: [],
            1: [],
            2: []
        }
        gammas = ["0", "1", "10", "100", "1000"]
        for gamma in values:
            for idx, alpha_val in enumerate(values[gamma]):
                dict_gammas[idx].append(alpha_val)

        plt_RMSE = plt.plot(gammas, dict_gammas[0], marker='o', linestyle='-',
                            label="RMSE optimised (Compositional Fairness Model)", color="#285e2d")
        plt_F1 = plt.plot(gammas, dict_gammas[1], marker='^', linestyle='-',
                          label="F1 optimised (Compositional Fairness Model)",
                          color="#46ab50")

        plt.ylabel(label_dict[el], fontsize=10)
        plt.yticks(fontsize=9)
        plt.xlabel("γ", fontsize=10)
        plt.xticks(fontsize=9)
        plt.title(dataset)
        plt.legend(loc="best", ncol=1, fontsize=8)
        #
        # plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    opt_measures = {
        'DiversityTradeoff': ["RMSE", "AD@20", "ID@20"],
        'FlexibleFairness': ["RMSE", "F1_age"]
    }

    function_map = {
        'plot_correlation_matrix': plot_correlation_matrix,
        'scatter_plot': scatter_plot,
        'plot_prediction_disagreement': plot_prediction_disagreement,
        'plot_averages': plot_averages,
        'plot_boxplot': plot_boxplot,
        'plot_alphas': plot_alpha_variants,
        'plot_gammas': plot_gamma_variants
    }

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ml100k')
    parser.add_argument("--command", choices=function_map.keys())
    args = parser.parse_args()

    args.command = 'plot_gammas'

    func = function_map[args.command]
    func(args.dataset, opt_measures)

# plot_alpha_variants()

# plot_gamma_variants()
