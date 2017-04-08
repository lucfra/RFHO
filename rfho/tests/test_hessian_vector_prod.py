import tensorflow as tf
from rfho.utils import hvp
# import tensorflow.python.ops.gradients_test
import numpy as np
import unittest


class TestHV(unittest.TestCase):

    def test_hvp(self):
        """
        Test for hessian vector product
        :return:
        """
        print('test 1')
        d = 20
        x = tf.Variable(tf.random_normal([d]))
        # noinspection PyTypeChecker
        fx = 3*tf.reduce_sum(x**3)
        vec = tf.Variable(tf.ones([d]))
        res = hvp(fx, x, vec)

        with tf.Session().as_default() as ss:

            ss.run(tf.global_variables_initializer())

            hessian = 18. * np.eye(d) * ss.run(x)
            self.assertLess(np.linalg.norm(
                ss.run(res) - hessian.dot(ss.run(vec))),
                1e-5
            )

    def test_hv_matrix(self):
        """
        Test for hessian vector product
        :return:
        """
        print('test 2')
        d = 20
        x = tf.Variable(tf.random_normal([d]))
        # noinspection PyTypeChecker
        fx = 3*tf.reduce_sum(x**3)
        vec = tf.Variable(tf.ones([d, 2]))
        res = tf.stack([
            hvp(fx, x, vec[:, k]) for k in range(vec.get_shape().as_list()[1])
            ], axis=1)

        with tf.Session().as_default() as ss:

            ss.run(tf.global_variables_initializer())

            hessian = np.eye(d) * ss.run(x) * 18.
            self.assertLess(np.linalg.norm(
                ss.run(res) - hessian.dot(ss.run(vec))),
                1e-5
            )


if __name__ == '__main__':
    unittest.main()
