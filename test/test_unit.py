import unittest
from math import sqrt
import pandas as pd
from src import get_samples_from_dataset, get_split, test_split, gini_index, gini_impurity

class CrossTestCase(unittest.TestCase):

    def rtest_get_samples_from_dataset(self):
        d = {'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
             'col2': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4]}
        df = pd.DataFrame(data=d)
        sets = get_samples_from_dataset(df, 2)
        self.assertEqual(len(sets), 2)
        self.assertEqual(len(sets[0]), 5)
        self.assertEqual(len(sets[0]), len(sets[1]))

    def rtest_get_samples_from_dataset_with_sample_size(self):
        d = {'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
             'col2': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4]}
        df = pd.DataFrame(data=d)
        sets = get_samples_from_dataset(df, 2, len(df))
        self.assertEqual(len(sets), 2)
        self.assertEqual(len(sets[0]), 10)
        self.assertEqual(len(sets[0]), len(sets[1]))

    def rtest_get_split(self):
        d = {
            'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'col2': [2, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col3': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col4': [4, 4, 3, 4, 3, 4, 3, 4, 3, 4]
        }
        df = pd.DataFrame(data=d)
        #result = get_split(df, len(df.columns))

    def rtest_test_split(self):
        d = {
            'col1': [1, 3, 1, 3, 2, 2, 1, 2, 1, 2],
            'col2': [2, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col3': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col4': [4, 4, 3, 4, 3, 4, 3, 4, 3, 4]
        }
        df = pd.DataFrame(data=d)
        left, right = test_split('col1', 2, df)
        self.assertEqual(len(left), 4)
        self.assertEqual(len(right), 6)

    def rtest_get_split_2(self):
        print("\n TESTING")
        d = {
            'col1': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            'col2': [2, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col3': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col4': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col5': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col6': [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            'col7': [4, 4, 3, 4, 3, 4, 3, 4, 3, 4]
        }
        df = pd.DataFrame(data=d)
        n_features = int(sqrt(len(df.columns)-1))
        features = get_split(df, n_features)

    def rtest_gini_index_even_split(self):
        print("\nTEST")
        df = [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3]
        ]
        left = [df[0], df[1]]
        right = [df[2], df[3]]
        groups = [left, right]
        class_values = list(set(row[-1] for row in df))
        index = gini_index(groups, class_values)
        self.assertEqual(index, 0.0)

    def test_gini_index_uneven_split(self):
        print("\nTEST INDEX")
        df = [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [3, 3, 1, 3, 3, 3],
            [3, 3, 1, 3, 3, 3]
        ]
        left = [df[0], df[1], df[2], df[3]]
        right = []
        groups = [left, right]
        class_values = list(set(row[-1] for row in df))
        index = gini_index(groups, class_values)
        self.assertEqual(index, 0.5)

    def rtest_gini_impurity_with_df_even_split(self):
        d = {
            'col1': [1, 1, 3, 3],
            'col2': [1, 1, 3, 3],
            'col3': [1, 1, 3, 3],
            'col4': [1, 1, 3, 3],
            'col5': [1, 1, 3, 3],
            'col6': [1, 1, 3, 3]
        }
        df = pd.DataFrame(data=d)
        groups = test_split('col3', 2, df)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 2)
        class_values = pd.unique(df[df.columns[-1]])
        index = gini_index(groups, class_values)
        print(index)
        self.assertEqual(index, 0.5)

    def test_gini_impurity_with_df_uneven_split(self):
        print("\nTEST IMPURITY")
        d = {
            'col1': [1, 1, 3, 3],
            'col2': [1, 1, 3, 3],
            'col3': [1, 1, 1, 1],
            'col4': [1, 1, 3, 3],
            'col5': [1, 1, 3, 3],
            'col6': [1, 1, 3, 3]
        }
        df = pd.DataFrame(data=d)
        groups = test_split('col3', 2, df)
        self.assertEqual(len(groups[0]), 4)
        self.assertEqual(len(groups[1]), 0)
        class_values = pd.unique(df[df.columns[-1]])
        index = gini_impurity(groups, class_values)
        print(index)
        self.assertEqual(0.5, index)



if __name__ == '__main__':
    unittest.main()