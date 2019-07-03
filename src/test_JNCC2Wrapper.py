from unittest import TestCase
from feutils import DataHandler
from feutils import JNCC2Wrapper
import tempfile
import numpy as np
import pandas as pd


class TestJNCC2Wrapper(TestCase):

    def test_fit(self):
        dh = DataHandler('__unittests', '__jncc2wrapper', 0)
        clf = JNCC2Wrapper(dh)

        print('params ' + str(clf.get_params()))

        # generate random binary classification task dataset
        X = np.random.rand(10, 3)
        y = np.random.randint(2, size=10)

        clf.fit(X, y)

        X = pd.DataFrame(X, columns=['a', 'b', 'c'])
        clf.fit(X, y)

    def test_predict(self):
        dh = DataHandler('__unittests', '__jncc2wrapper', 1)
        clf = JNCC2Wrapper(dh)

        # generate random binary classification task dataset
        X = np.random.rand(10, 3)
        y = np.random.randint(2, size=10)
        X = pd.DataFrame(X, columns=['a', 'b', 'c'])
        clf.fit(X, y)

        X_test = np.random.rand(5, 3)
        y_test = np.random.randint(2, size=5)
        X_test = pd.DataFrame(X_test, columns=['a', 'b', 'c'])
        res = clf.predict(X_test, y_test)
        print(res)
        assert(res.shape == (5,1))


    def test_score(self):
        dh = DataHandler('__unittests', '__jncc2wrapper', 2)
        clf = JNCC2Wrapper(dh)

        # generate random binary classification task dataset
        X = np.random.rand(10, 3)
        y = np.random.randint(2, size=10)
        X = pd.DataFrame(X, columns=['a', 'b', 'c'])
        clf.fit(X, y)

        X_test = np.random.rand(5, 3)
        y_test = np.random.randint(2, size=5)
        X_test = pd.DataFrame(X_test, columns=['a', 'b', 'c'])
        res = clf.score(X_test, y_test)
        print("FIT 1 score 1 " + str(res))

        X_test = np.random.rand(5, 3)
        y_test = np.random.randint(2, size=5)
        X_test = pd.DataFrame(X_test, columns=['a', 'b', 'c'])
        res = clf.score(X_test, y_test)
        print("FIT 1 score 2 " + str(res))

        X = np.random.rand(10, 3)
        y = np.random.randint(2, size=10)
        X = pd.DataFrame(X, columns=['a', 'b', 'c'])
        clf.fit(X, y)
        X_test = np.random.rand(5, 3)
        y_test = np.random.randint(2, size=5)
        X_test = pd.DataFrame(X_test, columns=['a', 'b', 'c'])
        res = clf.score(X_test, y_test)
        print("FIT 2 score 1 " + str(res))

    def test__predict_unknownclasses(self):
        dh = DataHandler('__unittests', '__jncc2wrapper', 3)
        clf = JNCC2Wrapper(dh)

        # generate random binary classification task dataset
        X = np.random.rand(10, 3)
        y = np.random.randint(2, size=10)
        X = pd.DataFrame(X, columns=['a', 'b', 'c'])
        clf.fit(X, y)

        X_test = np.random.rand(5, 3)
        y_test = None
        X_test = pd.DataFrame(X_test, columns=['a', 'b', 'c'])
        res = clf.predict(X_test, y_test)
        print(res)
