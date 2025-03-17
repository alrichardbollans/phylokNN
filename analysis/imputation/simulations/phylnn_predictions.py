from tqdm import tqdm

from analysis.imputation.helper_functions import phylnn_predict, number_of_simulation_iterations, missingness_types


def main():
    # Simulations
    for tag in tqdm(range(1, number_of_simulation_iterations+1)):
        for m in missingness_types:
            phylnn_predict('simulations', 'continuous', tag, m)

            phylnn_predict('simulations', 'binary', tag, m)


if __name__ == '__main__':
    main()
