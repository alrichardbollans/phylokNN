from tqdm import tqdm

from analysis.imputation.helper_functions import phylnn_predict, number_of_simulation_iterations, missingness_types


def main():
    for m in missingness_types:
        for tag in tqdm(range(1, number_of_simulation_iterations + 1)):
            phylnn_predict('simulations', 'continuous', tag, m)

            phylnn_predict('simulations', 'binary', tag, m)

            phylnn_predict('BMT', 'continuous', tag, m)
            phylnn_predict('EB', 'continuous', tag, m)

            phylnn_predict('BISSE', 'binary', tag, m)
            phylnn_predict('HISSE', 'binary', tag, m)

            phylnn_predict('Extinct_BMT', 'continuous', tag, m)
            phylnn_predict('Extinct_BMT', 'binary', tag, m)


if __name__ == '__main__':
    main()
