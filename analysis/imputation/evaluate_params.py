import os
import pickle

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis.imputation.evaluate_binary_outputs import evaluate_bin_output
from analysis.imputation.evaluate_continuous_outputs import evaluate_continuous_output
from analysis.imputation.phylnn_predictions import continuous_input_path, continuous_output_path, binary_output_path, binary_input_path


def evaluate_model_params(continuous:bool=True):
    lambdas = []
    ratio_max_dists = []
    gt_kappas = []
    model_kappas = []
    phylnn_scores = []
    if continuous:
        in_path = continuous_input_path
        out_path = continuous_output_path
    else:
        in_path = binary_input_path
        out_path = binary_output_path
    for t in range(1, 11):
        tag = str(t)
        lamba = pd.read_csv(os.path.join(in_path, tag, 'dataframe_params.csv'))['lambda'].iloc[0]
        kappa = pd.read_csv(os.path.join(in_path, tag, 'dataframe_params.csv'))['kappa'].iloc[0]
        fitted_gridsearch = pickle.load(open(os.path.join(out_path, tag, 'phylnn_hparams.pkl'), 'rb'))
        ratio = fitted_gridsearch.best_params_['ratio_max_branch_length']
        model_kappa = fitted_gridsearch.best_params_['kappa']
        lambdas.append(lamba)
        gt_kappas.append(kappa)
        ratio_max_dists.append(ratio)
        model_kappas.append(model_kappa)
        if continuous:

            phylnn_score, phylopars_score = evaluate_continuous_output(tag)
        else:
            phylnn_score, phylopars_score = evaluate_bin_output(tag)

        phylnn_scores.append(phylnn_score)
    if continuous:
        plot_path = os.path.join('plots', 'continuous')
    else:
        plot_path = os.path.join('plots', 'binary')
    sns.scatterplot(x=lambdas, y=ratio_max_dists)
    plt.savefig(os.path.join(plot_path, 'lambda_ratio_plot.jpg'), dpi=300)
    plt.close()
    sns.scatterplot(x=gt_kappas, y=model_kappas)
    plt.savefig(os.path.join(plot_path, 'kappa_plot.jpg'), dpi=300)
    plt.close()

    ax = sns.scatterplot(x=gt_kappas, y=phylnn_scores)
    ax.set_xlabel("Ground truth kappa'")
    ax.set_ylabel("PhyloNN loss")
    plt.savefig(os.path.join(plot_path, 'kappa_score_plot.jpg'), dpi=300)
    plt.close()

    ax = sns.scatterplot(x=lambdas, y=phylnn_scores)
    ax.set_xlabel("Ground truth lambda")
    ax.set_ylabel("PhyloNN loss")
    plt.savefig(os.path.join(plot_path, 'lambda_score_plot.jpg'), dpi=300)
    plt.close()


def summarise_model_params(tags):
    # plto distributions etc.
    pass


if __name__ == '__main__':
    evaluate_model_params()
    evaluate_model_params(False)
