# Master Thesis: Interdependency of Bias, Fairness and Prediction Instability in Graph-based Recommender Systems

This repository contains the code needed to run the experiments conducted in the master thesis "Interdependency of Bias, Fairness and Prediction Instability in Graph-based Recommender Systems".
The repository also includes the preprocessed MovieLens 100K dataset in addition to the optimised parameter settings for one instantiation for each optimisation objective as a starting point for the example pipeline.

The original datasets used in the thesis can be accessed here:
* https://grouplens.org/datasets/movielens/100k/
* https://grouplens.org/datasets/movielens/1m/
* https://github.com/ashwanidv100/Recommendation-System---Book-Crossing-Dataset/tree/master/BX-CSV-Dump (the header lines of the csv files need to be manually removed before using the dataset)




## Results
The main finding of this thesis is the cross-dataset and cross-recommender model strong relationship between individual diversity and popularity bias in recommender results.
This proves the existence of pronounced filter bubbles in the used datasets. A high individual diversity (each user being recommended a diverse selection of items) being related to a high level of popularity bias (more popular than unpopular items among the recommended items) indicates that there are thematically diverse filter bubbles, each containing a number of very popular items.
Further and more in-depth results as well as additional case studies concerning the used recommender models can be found in the thesis document. The following only depicts a very brief excerpt of the thesis.

 
Optimising on aggregated diversity or user fairness reduces the popularity bias of recommendation results while increasing the aggregated diversity, in comparison to the performance-optimised baseline..

<img src="https://github.com/nadine-ma/Thesis_interdependency_bias_fairness_instability_graph_recommenders/assets/72455501/ad35f2bd-5de7-42d8-89d7-ce060c34889f" width="70%">

Optimising on individual diversity decreases the aggregated diversity of recommendation results, while optimising on aggregated diversity or user fairness increases the level of aggregated diversity in recommendation results.

<img src="https://github.com/nadine-ma/Thesis_interdependency_bias_fairness_instability_graph_recommenders/assets/72455501/b69ad743-ecbe-4cdf-877a-e557f40e3e6a" width="70%">

Optimising on individual diversity only slightly increases the level of individual diversity in recommendation results, while optimising on aggregated diversity or user fairness decreases the level of individual diversity.

<img src="https://github.com/nadine-ma/Thesis_interdependency_bias_fairness_instability_graph_recommenders/assets/72455501/6da111e5-7056-4e7d-a1ee-63733c469b42" width="70%">

Optimising on either aggregated diversity or individual diversity increases the user unfairness in recommendation results, in comparison to the performance-optimised baseline. Optimising on user fairness decreases the user unfairness.

<img src="https://github.com/nadine-ma/Thesis_interdependency_bias_fairness_instability_graph_recommenders/assets/72455501/b7f5541b-efb1-431e-b0b0-87ac51cd8beb" width="70%">


The following correlation matrices depict the correlations between all obtained evaluation measures of all different optimisation objectives per dataset:

<img src="https://github.com/nadine-ma/Thesis_interdependency_bias_fairness_instability_graph_recommenders/assets/72455501/f5dc1b9e-d049-42b9-be5e-ad5be35060f7" width="70%">


While both used recommender models as well as the used datasets show differences regarding the levels of correlation between the analysed evaluation measures (cf. thesis p.38), they all show a very high, positive correlation between individual diversity (ID@20) and popularity bias (ARP@20) over all optimisation objectives.

<img src="https://github.com/nadine-ma/Thesis_interdependency_bias_fairness_instability_graph_recommenders/assets/72455501/f563f21d-3d0b-4604-be6c-d094d9a589ea" width="70%">






# Working with the Code
## Visual Exploration of the datasets

Parameters:
* `--dataset[str: choices = [ml100k, ml1m, bookCrosing]: default = ml100k]` defines the dataset.
* `--command[str]` defines the exploration function that should be used

The `--command` parameter can take on the following values.
* `plot_age_distribution` plots a bar graph showing how many users are in each age group.
* `plot_rating_distribution` plots a graph showing how the training ratings are distributed over the possible rating categories.
* `plot_rating_distribution_per_age` plots a graph showing how many training ratings users have given per rating category, divided by age catgories.
* `get_average_rating` returns the average rating of the training samples. 
* `get_average_ratings_per_age` returns the average rating for each age category of the training samples.
* `plot_ratings_per_user` plots the distribution of training ratings over all users
* `plot_longtail_distribution` plots the longtail distribution of the training ratings
* `plot_correlation_ratings` plots a scatter plot showing the correlation between item popularity and average rating
* `boxplot_item_popularity_by_age_group` plots a boxplot diagram showing the distribution of item popularity values rated by the different age groups


```
python dataset_exploration.py --dataset="ml100k" --command="plot_ratings_per_user"
```



## Running the example pipeline
The example pipeline offers the all-in-one code to use the provided MovieLens100K dataset settings. Due to varying training seeds for training additional model instantiations and based on the chosen epoch number, it is possible that the results obtained from running the example pipeline will not match the results presented in the thesis.

* RQ1: The pipeline trains x instantiations per optimisation objective, plots the average values for the individual evaluation measures and plots the variance of values using boxplots 
* RQ2: The pipeline visualises the prediction instability using these instantiations 
* RQ3: Correlation matrices is plotted using the calculated evaluation scores of each instantiation 
* Case study 2: The pipeline trains the Compositional Fairness model using varying values of fairness penalty and plots the results for the varying values 
* Case study 3: The pipeline trains the Diversity Tradeoff model using varying influences of the Furthest Neighbour graph and plots the results for the varying values


Parameters:
* `--dataset[str: choices = [ml100k, ml1m, bookCrosing]: default = ml100k]` defines the datasets that should be used 
* `--num_epochs[int: default = 10]` defines the number of training epochs
* `--num_instances[int: default = 2]` defines the number of additional instantiations to be trained per optimisation objectives

```
python example_pipeline.py --dataset="ml100k" --num_epochs=10 --num_instances=5
```






## Diversity Tradeoff model
implementation taken from https://github.com/esilezz/accdiv-via-graphconv
The Diversity Tradeoff model can also be run on its own outside of the example pipeline.

The parameters used by the Diversity Tradeoff model are as follows:

* `--data_dir[str: default = datasets]` specifies the folder containing the datasets.
* `--dataset[str: choices = [ml100k, ml1m, bookCrosing]: default = ml100k]` defines the dataset to use.
* `--num_random_items[int: default = 1000]` defines the number of items that will be used for the Value Unfairness (user unfairness) evaluation.
* `--batch_size[int: default = 128]` defines the batch size.
* `--eval_ratio[float: default = 0.1]` defines the ratio of all ratings that will be used for validation 
* `--test_ratio[float: default = 0.1]` defines the ratio of all ratings that will be used for testing. Training_ratio = 1 - eval_ratio - test_ratio
* `--top_k[int: default = 20]` defines the top-k recommendations per user, used for the ranking evaluation.
* `--lr[float: default = 0.01]` defines the learning rate.
* `--num_epochs[int: default = 50]` defines the number of epochs.
* `--seed[int: default = 0]` defines the training seed. A value of 0 leads to randomisation of the seed before training, other values will be used as they are specified.
* `--graph_type[str: IU (default), UI, UU, II]` defines the combination of users/items to use. The first letter defines the similarity source, while the second defines the dissimilarity source (U for users, I for Items).
* `--sim_order[int: default = 1]` defines the order of the graph filter applied on the similarity graph.
* `--dis_order[int: default = 1]` defines the order of the graph filter applied on the dissimilarity graph.
* `--alpha[float: default = 0.1]` defines the value of alpha. How much importance should be given to the dissimilarity graph. Scale from 0.1 - 0.9.
* `--mu[float: default = 0.5]` defines the parameter responsible for the overfitting in the optimization function.
* `--features[list: default = [1,2]]` defines the number of features of the GCNN.
* `--sim_users_NN[int: default = 30]` defines the number of neighbours in the user similarity graph.
* `--sim_items_NN[int: default = 35]` defines the number of neighbours in the item similarity graph.
* `--dis_NN[int: default = 40]` defines the number of neighbours in the dissimilarity graph.


To run the Diversity Tradeoff model execute the following:
```
python run_DiversityTradeoff.py --dataset="ml100k" --batch_size=128 --features="[1,2]"
```



## Flexible Fairness / Compositional Fairness model
implementation taken from https://github.com/joeybose/Flexible-Fairness-Constraints
The Flexible Fairness model can also be run on its own outside of the example pipeline.

The parameters used by the Flexible Fairness / Compositional Fairness model are as follows:

* `--data_dir[str: default = datasets]` specifies the folder containing the datasets.
* `--dataset[str: ml100k (default), ml1m, bookCrossing]` defines the dataset to use.
* `--num_random_items[int: default = 1000]` defines the number of items that will be used for the Value Unfairness (user unfairness) evaluation.
* `--batch_size[int: default = 128]` defines the batch size.
* `--eval_ratio[float: default = 0.1]` defines the ratio of all ratings that will be used for validation 
* `--test_ratio[float: default = 0.1]` defines the ratio of all ratings that will be used for testing. Training_ratio = 1 - eval_ratio - test_ratio
* `--top_k[int: default = 20]` defines the top-k recommendations per user, used for the ranking evaluation.
* `--lr[float: default = 0.01]` defines the learning rate.
* `--num_epochs[int: default = 50]` defines the number of epochs.
* `--seed[int: default = 0]` defines the training seed. A value of 0 leads to randomisation of the seed before training, other values will be used as they are specified.
* `--embed_dim[int: default = 10]` defines the embedding dimension for GCMC model
* `--D_steps[int: default = 10]` D_steps of age Discriminator model ToDo
* `--use_age_attr[bool: default = True]` defines whether the age attribute is used. Should always be True. Fairness objective can be still disabled via the gamma parameter.
* `--use_cross_entropy[bool: default = False]` Influence on age Discriminator calculation ToDO
* `--gamma[int: default = 0]` defines the value of gamma. How much fairness penalty should be applied during training. The higher gamma, the higher the fairness penalty.

To run the Diversity Tradeoff model execute the following:
```
python run_FlexibleFairness.py --dataset="ml100k" --batch_size=128 --gamma=10
```







## Running the optimisation after training the models
Parameter optimisation has to be finished before continuing to this step. No further model training to test parameters is possible after this step.
Evaluates all results to find the best performing instantiations and trains additional instantiations, gamma instantiations, alpha instantiations
Tests all optimised results

Parameters:
* `--dataset[str: default = ml100k]` defines the dataset that should be used
* `--model[str: choices: [FlexibleFairness, DiversityTradeoff]]` defines the exploration function that should be used
* `--num_instances[int: default = 10]` defines how many additional instantiations should be 


```
python get_optimisation.py --dataset="ml100k" --model="DiversityTradeoff" --num_instances=10
```



## Visual Exploration of the optimisation results
Optimisation has to be finished before running the visualisation of results. 

Parameters:
* `--dataset[str: choices = [ml100k, ml1m, bookCrosing]]` defines the dataset that should be used for evaluation
* `--command[str]` defines the exploration function that should be used

The `--command` parameter can take on the following values.
* `plot_correlation_matrix` plots a correlation matrix for every dataset and model combination.
* `scatter_plot` plots a scatter plot to visualise behaviour correlations between individual evaluation measures.
* `plot_prediction_disagreement` plots the prediction disagreement (instability).
* `plot_averages` plots the average results  for every evaluation measures.
* `plot_boxplot` plots the distribution of evaluation measure scores for the model instantiations using varying seeds via boxplot.



```
python plot_results.py --dataset="ml100k" --command="plot_ratings_per_user"
```















