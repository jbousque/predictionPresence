from unittest import TestCase
from feutils import DatasetHelper
import pandas as pd

class TestDatasetHelper(TestCase):

    def test_discretize_class(self):
        dsh = DatasetHelper()

        data_dict = {
            'feat1': [0.2, 0.1, 0.2, 0.5, 0.3, 0.2, 0., 0.4, 0.2, 0.3],
            'feat2' : [0.2, 0., 0.2, 0.5, 0.3, 0.2, 0., 0.4, 0.2, 0.],
            'Presence Score': [1.1, 2.3, 2.5, 3.2, 4.5, 4.2, 2.4, 1.5, 2.9, 3.3],
            'Co-presence Score': [1.3, 2., 2.1, 3.5, 4., 3.9, 1.9, 1.1, 3.2, 3.]
        }
        data = pd.DataFrame.from_dict(data_dict)

        class_col = dsh.discretize_class(
            data,
            prediction_type = 'classification',
            prediction_task = 'presence',
            bins = 2,
            strategy = 'kmeans'
        )

        print(class_col)
        print(data)

    def test_discretize_class_multioutput(self):
        dsh = DatasetHelper()

        data_dict = {
            'feat1': [0.2, 0.1, 0.2, 0.5, 0.3, 0.2, 0., 0.4, 0.2, 0.3],
            'feat2' : [0.2, 0., 0.2, 0.5, 0.3, 0.2, 0., 0.4, 0.2, 0.],
            'Presence Score': [1.1, 2.3, 2.5, 3.2, 4.5, 4.2, 2.4, 1.5, 2.9, 3.3],
            'Co-presence Score': [1.3, 2., 2.1, 0.2, 4., 3.9, 1.9, 1.1, 0.2, 1]
        }
        data = pd.DataFrame.from_dict(data_dict)

        class_col = dsh.discretize_class(
            data,
            prediction_type = 'classification',
            prediction_task = 'both',
            bins = 3,
            strategy = 'kmeans'
        )

        print(class_col)
        print(data)

        print(data[class_col])