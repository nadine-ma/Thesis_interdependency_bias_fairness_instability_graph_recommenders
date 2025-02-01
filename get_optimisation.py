import itertools
import os
from argparse import ArgumentParser
import random

import numpy as np

from run_FlexibleFairness import run_FlexibleFairness
from run_DiversityTradeoff import run_DiversityTradeoff
from test_FlexibleFairness import test_checkpoint_FlexibleFairness
from test_DiversityTradeoff import test_checkpoint_DiversityTradeoff


# creates text files containing the model and epoch returning the best result for each optimisation objective
def get_optimised_values(dataset, model, eval_measures):
    if model == "DiversityTradeoff":
        variance_parameter = "alpha"
    elif model == "FlexibleFairness":
        variance_parameter = "gamma"
    # eval_measures = ast.literal_eval(eval_measures)
    dict_eval_measures = {}
    dict_eval_measures_list = {}
    for el in eval_measures:
        dict_eval_measures_list[el] = []
        dict_eval_measures[el] = {}

    data_model_dir = os.path.join('logs', dataset, model)
    for dir_name in os.listdir(data_model_dir):
        run_dir = os.path.join(data_model_dir, dir_name)
        if os.path.isdir(run_dir):
            best_epochs = {}
            variance_parameter_value = 0

            with open(os.path.join(run_dir, 'best_epochs.txt')) as f:
                for idx, line in enumerate(f):
                    parts = line.strip('\n').split(",")
                    if parts[1] not in best_epochs:
                        best_epochs[parts[1]] = []
                    best_epochs[parts[1]].append(parts[0])  # in epoch x is the best value for measure y

            with open(os.path.join(run_dir, 'hparams.txt')) as f:
                for idx, line in enumerate(f):
                    parts = line.strip('\n').split(":")
                    if parts[0] == variance_parameter:
                        variance_parameter_value = parts[1]



            with open(os.path.join(run_dir, 'eval_measures.csv')) as f:
                for epoch, line in enumerate(f):
                    epoch = str(epoch)
                    if epoch in list(best_epochs.keys()):
                        opt_measures = best_epochs[epoch]
                        parts = line.strip('\n').split(",")
                        for part in parts:
                            opt_measure = part.split(":")[0]
                            if opt_measure in opt_measures:
                                opt_measure_value = part.split(":")[1]
                                dict_eval_measures[opt_measure][opt_measure_value] = (run_dir, epoch, variance_parameter_value)
                                dict_eval_measures_list[opt_measure].append(opt_measure_value)
    dict_best_val = {
        'RMSE': 'min',
        'AD@20': 'max',
        'ID@20': 'max',
        'ARP@20': 'min',
        'RMSE_coeff_var': 'min',
        'VU_coeff_var': 'min',
        'F1_age': 'min'
    }

    if not os.path.exists(os.path.join(data_model_dir, 'optimised_epochs.txt')):
        with open(os.path.join(data_model_dir, 'optimised_epochs.txt'), 'a') as file:
            for opt_measure in eval_measures:
                best_val = 0
                if dict_best_val[opt_measure] == "max":
                    best_val = max(dict_eval_measures_list[opt_measure])
                else:
                    best_val = min(dict_eval_measures_list[opt_measure])
                best_run_dir, best_epoch, variance_parameter_value = dict_eval_measures[opt_measure][best_val]
                file.write(f'{opt_measure},{best_run_dir},{best_epoch},{variance_parameter_value}\n')

    types = ["instances","alphas"]
    if model == "FlexibleFairness":
        types = ["instances", "gammas"]
    for opt_measure in eval_measures:

        for type in types:
            if not os.path.exists(os.path.join(data_model_dir, f'{type}_{opt_measure}.txt')):
                with open(os.path.join(data_model_dir, f'{type}_{opt_measure}.txt'), 'a') as file:
                    best_val = 0
                    if dict_best_val[opt_measure] == "max":
                        best_val = max(dict_eval_measures_list[opt_measure])
                    else:
                        best_val = min(dict_eval_measures_list[opt_measure])
                    best_run_dir, best_epoch, variance_parameter_value = dict_eval_measures[opt_measure][best_val]
                    if type == "instances":
                        file.write(f'{best_run_dir}\n')
                    if type in ["alphas", "gammas"]:
                        file.write(f'{best_run_dir},{variance_parameter_value}\n')
    print()

# runs additional model instantiations using the parameter settings of the optimised instantiation for each optimisation objective
def run_instances(dataset, model, opt_measures, num_add_instances=1):
    parser = ArgumentParser()
    args = parser.parse_args()

    data_model_dir = os.path.join('logs', dataset, model)
    for opt_measure in opt_measures:
        run_dir = ""
        with open(os.path.join(data_model_dir, 'optimised_epochs.txt')) as f:
            for line in f:
                if line.split(",")[0] == opt_measure:
                    run_dir = line.split(",")[1]

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
        args.seed = random.randint(00000, 99999)
        for i in range(num_add_instances):
            if model == "DiversityTradeoff":
                run_DiversityTradeoff(args, instance_bool=opt_measure, alpha_bool=False)
            elif model == "FlexibleFairness":
                run_FlexibleFairness(args, instance_bool=opt_measure, gamma_bool=False)

# runs alpha variants for the original optimised objective instantiatons of the Diversity Tradeoff model
def run_alpha_variants(dataset, model, opt_measures):
    parser = ArgumentParser()
    args = parser.parse_args()

    data_model_dir = os.path.join('logs', dataset, model)
    for opt_measure in opt_measures:
        run_dir = ""
        with open(os.path.join(data_model_dir, 'optimised_epochs.txt')) as f:
            for line in f:
                if line.split(",")[0] == opt_measure:
                    run_dir = line.split(",")[1]

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
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        alphas.remove(args.alpha)

        for alpha in alphas:
            args.alpha = alpha
            run_DiversityTradeoff(args, instance_bool=False, alpha_bool=opt_measure)


# runs gamma variants for the original optimised objective instantiatons of the Flexible Fairness model
def run_gamma_variants(dataset, model, opt_measures):
    parser = ArgumentParser()
    args = parser.parse_args()

    data_model_dir = os.path.join('logs', dataset, model)
    for opt_measure in opt_measures:
        run_dir = ""
        with open(os.path.join(data_model_dir, 'optimised_epochs.txt')) as f:
            for line in f:
                if line.split(",")[0] == opt_measure:
                    run_dir = line.split(",")[1]

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
        gammas = [0,1,10,100,1000]
        gammas.remove(args.gamma)

        for gamma in gammas:
            args.gamma = gamma
            run_FlexibleFairness(args, instance_bool=False, gamma_bool=opt_measure)


# runs model testing for instantiations, alpha instantiations,  gamma instantiations
def get_optimised_tested(dataset, model, opt_measures, instance_bool=False, alpha_bool=False, gamma_bool=False):
    for opt_measure in opt_measures:
        data_model_dir = os.path.join('logs', dataset, model)
        run_dirs = []
        if instance_bool:
            pred_diffs = []
            with open(os.path.join(data_model_dir, f'instances_{opt_measure}.txt')) as f:
                for line in f:
                    run_dirs.append(line.strip("\n"))
            for run_dir in run_dirs:
                with open(os.path.join(run_dir, f'best_epochs.txt')) as f:
                    for line in f:
                        if line.split(",")[0] == opt_measure:
                            epoch = int(line.split(",")[1].strip("\n"))
                if model == "DiversityTradeoff":
                    preds = test_checkpoint_DiversityTradeoff(run_dir, opt_measure, epoch,
                                                              instance_bool=instance_bool,
                                                              alpha_bool=alpha_bool)
                    pred_diffs.append([round(pred) for pred in preds])
                if model == "FlexibleFairness":
                    preds = test_checkpoint_FlexibleFairness(run_dir, opt_measure, epoch,
                                                             instance_bool=instance_bool,
                                                             gamma_bool=gamma_bool)
                    pred_diffs.append([round(pred) for pred in preds])

            ######################## Prediction Instability ###########################
            diff = 0
            diffs = []
            combinations = list(itertools.combinations(pred_diffs, 2))
            for combi in combinations:
                for idx, el in enumerate(combi[0]):
                    if combi[0][idx] == combi[1][idx]:
                        diffs.append(0)
                    else:
                        diffs.append(1)
            pred_diff = np.mean(diffs)
            print(f"Prediction Disagreement: {pred_diff}")
            with open(os.path.join(data_model_dir, f'prediction_disagreements.txt'), 'a') as file:
                file.write(f"{opt_measure}:{pred_diff}\n")


        elif alpha_bool:
            with open(os.path.join(data_model_dir, f'alphas_{opt_measure}.txt')) as f:
                for line in f:
                    parts = line.strip("\n").split(",")
                    run_dirs.append(parts[0])

            for run_dir in run_dirs:
                with open(os.path.join(run_dir, f'best_epochs.txt')) as f:
                    for line in f:
                        if line.split(",")[0] == opt_measure:
                            epoch = int(line.split(",")[1].strip("\n"))
                if model == "DiversityTradeoff":
                    test_checkpoint_DiversityTradeoff(run_dir, opt_measure, epoch, instance_bool=instance_bool, alpha_bool=alpha_bool)

        elif gamma_bool:
            with open(os.path.join(data_model_dir, f'gammas_{opt_measure}.txt')) as f:
                for line in f:
                    parts = line.strip("\n").split(",")
                    run_dirs.append(parts[0])

            for run_dir in run_dirs:
                with open(os.path.join(run_dir, f'best_epochs.txt')) as f:
                    for line in f:
                        if line.split(",")[0] == opt_measure:
                            epoch = int(line.split(",")[1].strip("\n"))
                if model == "FlexibleFairness":
                    test_checkpoint_FlexibleFairness(run_dir, opt_measure, epoch, instance_bool=instance_bool, gamma_bool=gamma_bool)
    print()



if __name__ == '__main__':




    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--model", type=str, default="DiversityTradeoff")
    parser.add_argument("--num_instances", type=int, default=10)
    args = parser.parse_args()

    if args.model == "DiversityTradeoff":
        opt_measures = ['RMSE','AD@20','ID@20']
    elif args.model == "FlexibleFairness":
        opt_measures = ['RMSE', 'F1_age']

    get_optimised_values(args.dataset, args.model,opt_measures)
    run_instances(args.dataset,  args.model, opt_measures, num_add_instances=args.num_instances)
    get_optimised_tested(args.dataset,  args.model, opt_measures, instance_bool=True, alpha_bool=False)

    if args.model == "FlexibleFairness":
        run_gamma_variants(args.dataset, args.model, opt_measures)
        get_optimised_tested(args.dataset, args.model, opt_measures, instance_bool=False, alpha_bool=False,
                             gamma_bool=True)
    elif args.model == "DiversityTradeoff":
        run_alpha_variants(args.dataset, args.model, opt_measures)
        get_optimised_tested(args.dataset, args.model, opt_measures, instance_bool=False, alpha_bool=True,
                             gamma_bool=False)





