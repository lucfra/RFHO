import unittest

from rfho.utils import as_list

from rfho.optimizers import *
from tests.test_base import iris_logistic_regression


class TestOptimizers(unittest.TestCase):

    def _test_forward_automatic_d_dynamic_d_hyper(self, method, optimizer_hypers=None,
                                                  **opt_kwargs):
        iris, x, y, model, w, out, error, accuracy = iris_logistic_regression(
            method.get_augmentation_multiplier())

        shape_w = w.get_shape()

        scalar_hyper = tf.Variable(1., name='scalar_hyper')
        vector_hyper = tf.Variable(tf.ones([3]), name='vector_hyper')

        tr_err = tf.identity(error + tf.reduce_sum(w ** 2) * scalar_hyper
                             + vector_hyper * tf.stack([tf.reduce_sum(w.tensor[:5]),
                                                        tf.reduce_sum(w.tensor[5:10]),
                                                        tf.reduce_sum(w.tensor[10:])]),
                             name='training_error')
        optimizer = method.create(w, loss=tr_err, **opt_kwargs)

        d_phi_d_scalar_hyper = optimizer.auto_d_dynamics_d_hyper(scalar_hyper)
        self.assertIsNotNone(d_phi_d_scalar_hyper.tensor)
        self.assertListEqual(d_phi_d_scalar_hyper.get_shape().as_list(), [shape_w[0].value, 1])

        d_phi_d_vector_hyper = optimizer.auto_d_dynamics_d_hyper(vector_hyper)
        self.assertIsNotNone(d_phi_d_vector_hyper.tensor)
        self.assertListEqual(d_phi_d_vector_hyper.get_shape().as_list(),
                             [shape_w[0].value, vector_hyper.get_shape()[0].value])

        if optimizer_hypers:
            [self.assertIsNotNone(optimizer.auto_d_dynamics_d_hyper(hyp)) for hyp in as_list(optimizer_hypers)]

    def test_forward_automatic_d_dynamic_d_hyper(self):
        optimizer = GradientDescentOptimizer
        eta = tf.Variable(.1, name='eta')
        self._test_forward_automatic_d_dynamic_d_hyper(optimizer, optimizer_hypers=[eta], lr=eta)

        tf.reset_default_graph()

        optimizer = MomentumOptimizer
        eta = tf.Variable(.1, name='eta')
        mu = tf.Variable(0.9, name='mu')
        self._test_forward_automatic_d_dynamic_d_hyper(optimizer, optimizer_hypers=[eta, mu], lr=eta, mu=mu)


if __name__ == '__main__':
    unittest.main()
