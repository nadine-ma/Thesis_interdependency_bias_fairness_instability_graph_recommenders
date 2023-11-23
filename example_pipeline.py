import os
from argparse import ArgumentParser

import run_DiversityTradeoff
import run_FlexibleFairness
import get_optimisation
import plot_results

def load_example_parameters(dataset, model, measure):
    parser = ArgumentParser()
    hparams = parser.parse_args()
    path_to_settings = os.path.join("datasets", dataset, "example_parameters", f"{model}_example_parameters_{measure}.txt")
    with open(path_to_settings) as f:
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
    return hparams


def run_pipeline(hparams):
    '''
    The example pipeline is only for a quick visualisation of the whole experimental pipeline.
    We train 5 instantiations per model, with a few varying parameter settings.

    After, we find the optimised model checkpoint for each recommendation objective and train 4 additional
    instantiations using identical settings but a varying parameter.

    In the end we visualise the results using correlation matrices, bar plots etc.
    '''

    'Run Diversity Tradeoff model with varying parameters'
    args = load_example_parameters(hparams.dataset, 'DiversityTradeoff', "RMSE")
    run_DiversityTradeoff.run_DiversityTradeoff(args)
    args = load_example_parameters(hparams.dataset, 'DiversityTradeoff', "AD")
    run_DiversityTradeoff.run_DiversityTradeoff(args)
    args = load_example_parameters(hparams.dataset, 'DiversityTradeoff', "ID")
    run_DiversityTradeoff.run_DiversityTradeoff(args)


    'Run Flexible Fairness model with varying parameters'
    args = load_example_parameters(hparams.dataset, 'FlexibleFairness', "RMSE")
    run_FlexibleFairness.run_FlexibleFairness(args)
    args = load_example_parameters(hparams.dataset, 'FlexibleFairness', "F1")
    run_FlexibleFairness.run_FlexibleFairness(args)

    # #################### Diversity Tradeoff - get results####################

    model = "DiversityTradeoff"
    opt_measures = ['RMSE', 'AD@20', 'ID@20']

    'Search for optimised result for every opt_measure for the Diversity Tradeoff model'
    get_optimisation.get_optimised_values(hparams.dataset, model, opt_measures)

    'Create 4 additional instances for each opt_measure only changing the training seed'
    get_optimisation.run_instances(hparams.dataset, model, opt_measures, num_add_instances=hparams.num_instances)

    'Run testing for all optimised instances'
    get_optimisation.get_optimised_tested(hparams.dataset, model, opt_measures, instance_bool=True, alpha_bool=False)

    get_optimisation.run_alpha_variants(hparams.dataset, model, opt_measures)
    get_optimisation.get_optimised_tested(hparams.dataset,  model, opt_measures, instance_bool=False, alpha_bool=True, gamma_bool=False)

    # #################### Flexible Fairness - get results####################

    model = "FlexibleFairness"
    opt_measures = ['RMSE', 'F1_age']

    'Search for optimised result for every opt_measure for the Flexible Fairness model'
    get_optimisation.get_optimised_values(hparams.dataset, model, opt_measures)

    'Create 4 additional instances for each opt_measure only changing the training seed'
    get_optimisation.run_instances(hparams.dataset, model, opt_measures, num_add_instances=hparams.num_instances)

    'Run testing for all optimised instances'
    get_optimisation.get_optimised_tested(hparams.dataset, model, opt_measures, instance_bool=True,
                                          alpha_bool=False)
    get_optimisation.run_gamma_variants(hparams.dataset, model, opt_measures)
    get_optimisation.get_optimised_tested(hparams.dataset,  model, opt_measures, instance_bool=False, alpha_bool=False, gamma_bool=True)

    # ################################## Plot Results ##################################

    opt_measures = {
        'DiversityTradeoff': ["RMSE", "AD@20", "ID@20"],
        'FlexibleFairness': ["RMSE", "F1_age"]
    }
    plot_results.plot_correlation_matrix(hparams.dataset, opt_measures)
    plot_results.scatter_plot(hparams.dataset, opt_measures)
    plot_results.plot_averages(hparams.dataset, opt_measures)
    plot_results.plot_boxplot(hparams.dataset, opt_measures)
    plot_results.plot_prediction_disagreement(hparams.dataset, opt_measures)
    plot_results.plot_alpha_variants(hparams.dataset, opt_measures)
    plot_results.plot_gamma_variants(hparams.dataset, opt_measures)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_instances", type=int, default=2)
    args = parser.parse_args()
    run_pipeline(args)
