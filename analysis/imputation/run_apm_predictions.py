from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from analysis.imputation.helper_functions import missingness_types, get_prediction_data_paths, phylnn_predict
from analysis.imputation.run_encodings_predictions import get_umap_data, add_y_to_data, get_eigenvectors, get_autoencoded_data, logit_init_kwargs, \
    logit_grid_search_params, fit_and_output, xgb_clf_init_kwargs, xgb_clf_grid_search_params, get_semi_supervised_umap_data, \
    get_semi_supervised_autoencoded_data


def run_predictions():
    iteration = 1
    for m in missingness_types:
        bin_or_cont = 'binary'

        real_or_sim = 'my_apm_data'

        # phylnn_predict(real_or_sim, 'binary', 1, m)
        #
        # umap_X = get_umap_data(real_or_sim, bin_or_cont, iteration)
        # umap_df, umap_encoding_vars, umap_target_name = add_y_to_data(umap_X, real_or_sim, bin_or_cont, iteration, m)
        #
        # eigen_X = get_eigenvectors(real_or_sim, bin_or_cont, iteration)
        # eigen_df, eigen_encoding_vars, eigen_target_name = add_y_to_data(eigen_X, real_or_sim, bin_or_cont, iteration, m)
        #
        # autoenc_X = get_autoencoded_data(real_or_sim, bin_or_cont, iteration)
        # autoenc_df, autoenc_encoding_vars, autoenc_target_name = add_y_to_data(autoenc_X, real_or_sim, bin_or_cont, iteration, m)
        out_dir = get_prediction_data_paths(real_or_sim, bin_or_cont, iteration, m)

        if bin_or_cont == 'binary':
            # # Compare logistic regression and XGBoost models i.e. for modelling simpler relationships and complex relationships
            # clf_instance = LogisticRegression(**logit_init_kwargs)
            # fit_and_output(clf_instance, logit_grid_search_params, out_dir, 'logit_umap', umap_df, umap_encoding_vars, umap_target_name,
            #                bin_or_cont)
            #
            # clf_instance = LogisticRegression(**logit_init_kwargs)
            # fit_and_output(clf_instance, logit_grid_search_params, out_dir, 'logit_eigenvecs', eigen_df, eigen_encoding_vars,
            #                eigen_target_name, bin_or_cont)
            #
            # clf_instance = XGBClassifier(**xgb_clf_init_kwargs)
            # fit_and_output(clf_instance, xgb_clf_grid_search_params, out_dir, 'xgb_umap', umap_df, umap_encoding_vars, umap_target_name,
            #                bin_or_cont)
            #
            # clf_instance = XGBClassifier(**xgb_clf_init_kwargs)
            # fit_and_output(clf_instance, xgb_clf_grid_search_params, out_dir, 'xgb_eigenvecs', eigen_df, eigen_encoding_vars,
            #                eigen_target_name, bin_or_cont)
            #
            # # ### Semisupervised umap
            semi_supervised_umap_df, semi_sup_umap_encoding_vars, semi_sup_umap_target_name = get_semi_supervised_umap_data(real_or_sim,
                                                                                                                            bin_or_cont,
                                                                                                                            iteration, m)
            clf_instance = LogisticRegression(**logit_init_kwargs)
            fit_and_output(clf_instance, logit_grid_search_params, out_dir, 'logit_umap_supervised', semi_supervised_umap_df,
                           semi_sup_umap_encoding_vars,
                           semi_sup_umap_target_name, bin_or_cont)
            clf_instance = XGBClassifier(**xgb_clf_init_kwargs)
            fit_and_output(clf_instance, xgb_clf_grid_search_params, out_dir, 'xgb_umap_supervised', semi_supervised_umap_df,
                           semi_sup_umap_encoding_vars,
                           semi_sup_umap_target_name, bin_or_cont)

            # ### autoencoder
            # clf_instance = LogisticRegression(**logit_init_kwargs)
            # fit_and_output(clf_instance, logit_grid_search_params, out_dir, 'logit_autoencoded', autoenc_df, autoenc_encoding_vars,
            #                eigen_target_name, bin_or_cont)
            #
            # clf_instance = XGBClassifier(**xgb_clf_init_kwargs)
            # fit_and_output(clf_instance, xgb_clf_grid_search_params, out_dir, 'xgb_autoencoded', autoenc_df, autoenc_encoding_vars,
            #                umap_target_name,
            #                bin_or_cont)

            ## Semisupervised autoenc
            semi_supervised_autoenc_df, semi_sup_autoenc_encoding_vars, semi_sup_autoenc_target_name = get_semi_supervised_autoencoded_data(
                real_or_sim,
                bin_or_cont,
                iteration, m)

            clf_instance = LogisticRegression(**logit_init_kwargs)
            fit_and_output(clf_instance, logit_grid_search_params, out_dir, 'logit_autoenc_supervised', semi_supervised_autoenc_df,
                           semi_sup_autoenc_encoding_vars,
                           semi_sup_autoenc_target_name, bin_or_cont)
            clf_instance = XGBClassifier(**xgb_clf_init_kwargs)
            fit_and_output(clf_instance, xgb_clf_grid_search_params, out_dir, 'xgb_autoenc_supervised', semi_supervised_autoenc_df,
                           semi_sup_autoenc_encoding_vars,
                           semi_sup_autoenc_target_name, bin_or_cont)


if __name__ == '__main__':
    run_predictions()
