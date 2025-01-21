import unittest

import numpy as np
import pandas as pd

from phyloKNN import get_first_column


class test_arrays(unittest.TestCase):


    def test_types(self):
        X = [['A', 2, 3], ['B', 5, 6], ['C', 8, 9]]

        assert get_first_column(X) == ['A', 'B', 'C']

        df = pd.DataFrame(X)
        assert get_first_column(df) == ['A', 'B', 'C']

        num = np.array(X)
        assert get_first_column(num) == ['A', 'B', 'C']

        self.assertRaises(AssertionError, get_first_column, [[1, 2, 3], ['B', 5, 6], ['C', 8, 9]])

if __name__ == '__main__':
    unittest.main()
