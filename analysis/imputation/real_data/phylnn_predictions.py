from analysis.imputation.helper_functions import phylnn_predict


def main():
        phylnn_predict('real_data', 'continuous', 1, 'mcar')

        phylnn_predict('real_data', 'binary', 1, 'mcar')


if __name__ == '__main__':
    main()
