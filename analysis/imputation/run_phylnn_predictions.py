from tqdm import tqdm

from analysis.imputation.helper_functions import phylnn_predict, number_of_simulation_iterations, missingness_types


def main():
    # Simulations
    for m in missingness_types:
        for tag in tqdm(range(1, number_of_simulation_iterations+1)):

                phylnn_predict('simulations', 'continuous', tag, m)

                phylnn_predict('simulations', 'binary', tag, m)

    # real data
    phylnn_predict('real_data', 'continuous', 1, 'mcar')
    #
    # phylnn_predict('real_data', 'binary', 1, 'mcar')
if __name__ == '__main__':
    main()
