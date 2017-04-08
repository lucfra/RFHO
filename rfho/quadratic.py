import cvxopt
import numpy as np
from cvxopt import spmatrix, matrix, sparse

options = {'show_progress': False}
mode_strings = {1: 'keep the lambdas in [0,1] and  \sum{lambdas} < dim for the entire training set',
                2: 'keep the lambdas in [0,1] and  \sum{lambdas} < 1 for each example',
                3: 'keep the lambdas in [0,1] and  \sum{lambdas} = 1 for each example'}


class Projection:
    """Projection handler with the cvxopt library"""

    def __init__(self, dim, n_labels, mode=1):
        self.mode = mode
        self.dim = dim
        self.n_labels = n_labels
        if mode == 1:
            self.P = spmatrix(1, range(dim * n_labels), range(dim * n_labels))

            glast = matrix(np.ones((1, dim * n_labels)))
            self.G = sparse([-self.P, self.P, glast])

            h1 = np.zeros(dim * n_labels)
            h2 = np.ones(dim * n_labels)
            self.h = matrix(np.concatenate([h1, h2, [dim]]))
        elif mode == 2:
            self.P = spmatrix(1, range(n_labels), range(n_labels))
            glast = matrix(np.ones((1, n_labels)))
            self.G = sparse([-self.P, self.P, glast])

            h1 = np.zeros(n_labels)
            h2 = np.ones(n_labels)
            self.h = matrix(np.concatenate([h1, h2, [1]]))
        elif mode == 3:
            self.P = spmatrix(1, range(n_labels), range(n_labels))
            self.A = matrix(np.ones((1, n_labels)))
            self.G = sparse([-self.P, self.P])

            h1 = np.zeros(n_labels)
            h2 = np.ones(n_labels)
            self.h = matrix(np.concatenate([h1, h2]))
            self.b = matrix(np.ones(1))

    def run(self, x):
        print("start projection: {}".format(mode_strings[self.mode]))
        _val = None
        if self.mode == 1:
            _val = self._run1(x)
        if self.mode == 2:
            _val = self._run2(x)
        if self.mode ==3:
            _val = self._run3(x)
        print("end projection")
        return _val

    def _run1(self, x):
        # keep the lambdas in [0,1] and  \sum{lambdas} < dim for the entire training set
        shape = x.shape
        vec_x = x.reshape(-1)
        self.q = matrix(- np.array(vec_x, dtype=np.float64))
        _res = cvxopt.solvers.qp(self.P, self.q, self.G, self.h, initvals=self.q, options=options)
        _res2 = np.array(_res['x'], dtype=np.float32)[:, 0]
        return _res2.reshape(shape)

    def _run2(self, x):
        # keep the lambdas in [0,1] and  \sum{lambdas} < 1 for each example
        _res2 = np.empty((self.dim, self.n_labels))
        for i in range(x.shape[0]):
            # print("start example {}".format(i))
            self.q = matrix(- np.array(x[i], dtype=np.float64))
            _res = cvxopt.solvers.qp(self.P, self.q, self.G, self.h, initvals=self.q, options=options)
            _res2[i] = np.array(_res['x'], dtype=np.float32)[:, 0]
        return _res2

    def _run3(self, x):
        # keep the lambdas in [0,1] and  \sum{lambdas} = 1 for each example
        _res2 = np.empty((self.dim, self.n_labels))
        for i in range(x.shape[0]):
            # print("start example {}".format(i))
            self.q = matrix(- np.array(x[i], dtype=np.float64))
            _res = cvxopt.solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b, initvals=self.q, options=options)
            _res2[i] = np.array(_res['x'], dtype=np.float32)[:, 0]
        return _res2


def one_hot_accuracy(vec, epsilon):
    n_examples = vec.shape[0]
    near_one_hot = np.amax(vec, 1) > 1 - epsilon
    # print("near_one_hot ({}/{}): {}".format(near_one_hot.size, n_examples, near_one_hot))
    return np.count_nonzero(near_one_hot)/n_examples



# TODO: tests for the various projections







