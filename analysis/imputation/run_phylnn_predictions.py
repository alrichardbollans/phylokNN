from tqdm import tqdm

from analysis.data.helper_functions import number_of_simulation_iterations
from analysis.imputation.helper_functions import phylnn_predict, missingness_types


def main():
    for m in missingness_types:
        for tag in tqdm(range(1, number_of_simulation_iterations + 1)):
            phylnn_predict('simulations', 'continuous', tag, m)
            phylnn_predict('simulations', 'binary', tag, m)

            phylnn_predict('real_data', 'continuous', tag, m)
            phylnn_predict('real_data', 'binary', tag, m)

            phylnn_predict('BMT', 'continuous', tag, m)
            phylnn_predict('EB', 'continuous', tag, m)

            phylnn_predict('BISSE', 'binary', tag, m)
            phylnn_predict('HISSE', 'binary', tag, m)

            phylnn_predict('Extinct_BMT', 'continuous', tag, m)
            phylnn_predict('Extinct_BMT', 'binary', tag, m)


if __name__ == '__main__':
    main()
