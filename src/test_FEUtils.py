from unittest import TestCase
from feutils import FEUtils
import numpy as np

class TestFEUtils(TestCase):
    def test_compute_mrr(self):
        feu = FEUtils()
        y_true = np.array([0,1,1,0,0,1])
        y_pred = np.array([[0,666], [1,666], [0,666], [1,0], [0,666], [0,1]])
        mrr = feu.compute_mrr(y_true, y_pred, [0,1])
        print(mrr)
        assert mrr == 0.75