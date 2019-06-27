from unittest import TestCase
from feutils import FEUtils
import numpy as np
import pandas as pd

class TestFEUtils(TestCase):
    def test_compute_mrr(self):
        feu = FEUtils()
        y_true = np.array([0, 1])
        y_pred = np.array([[0, 666], [1, 666]])
        mrr = feu.compute_mrr(y_true, y_pred, [0, 1])
        print(mrr)
        assert mrr == 1

        feu = FEUtils()
        y_true = np.array([0, 1])
        y_pred = np.array([[1, 0], [1, 666]])
        mrr = feu.compute_mrr(y_true, y_pred, [0, 1])
        print(mrr)
        assert mrr == 0.75

    def test_compute_mrr_determined(self):
        feu = FEUtils()
        df = pd.DataFrame(np.random.randint(2,size=(10,2)), columns=['feat1', 'y'])
        y_true = df['feat1']
        y_pred = df['y']
        mrr = feu.compute_mrr(y_true, y_pred, [0, 1])
        print(mrr)
        assert mrr == 1


        """compute_mrr(y_true 6     1
48    1
7     1
34    1
51    1
53    0
50    1
60    0
3     0
Name: PresenceClass_2_uniform, dtype: int32, y_pred 0    1
1    1
2    1
3    0
4    1
5    0
6    1
7    1
8    0
Name: 7, dtype: int64, labels [0, 1]
(9,)"""