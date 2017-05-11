import unittest
import tensorflow as tf
from tests.test_base import iris_logistic_regression


class TestModelVectorization(unittest.TestCase):

    def test_augmented_gradient(self):
        iris, x, y, model, w, model_y, error, accuracy = iris_logistic_regression(2)

        d = int(w.tensor.get_shape()[0].value/3)
        self.assertEqual(d, 4*3 + 3)

        gt = tf.gradients(error, w.tensor)
        print('gradients w.r.t. augmented state', gt, sep='\n')
        self.assertFalse(any([g is None for g in gt]))  # all gradients are defined (all(gt) rises a tf error

        grads_intermediate = tf.gradients(error, w._var_list_as_tensors())
        print('gradients w.r.t. w, m, p', grads_intermediate, sep='\n')
        self.assertFalse(any([g is None for g in grads_intermediate]))

        grads_wrt_single_variables = tf.gradients(error, w._get_base_variable_list())
        print('gradients w.r.t. w, m, p', grads_wrt_single_variables, sep='\n')
        self.assertFalse(any([g is None for g in grads_wrt_single_variables]))

        fd = {x: iris.train.data, y: iris.train.target}
        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()

            print('gradients w.r.t. augmented state', ss.run(gt, feed_dict=fd), sep='\n')

            print('gradients w.r.t. w, m, p', ss.run(grads_intermediate, feed_dict=fd), sep='\n')

            print('gradients w.r.t. w, m, p', ss.run(grads_wrt_single_variables, feed_dict=fd), sep='\n')


if __name__ == '__main__':
    unittest.main()
