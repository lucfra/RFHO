import unittest
from tests.test_base import iris_logistic_regression
from rfho.models import *


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


class TestModels(unittest.TestCase):

    def test_sparse_input_models(self):
        import numpy as np
        import rfho.datasets as ddt

        real_sim = ddt.load_realsim(partitions_proportions=[.5, .3])

        model_train = LinearModel(real_sim.train.data, real_sim.train.dim_data, real_sim.train.dim_target,
                                   init_w=tf.random_normal, init_b=tf.random_normal, benchmark=True)
        model_train2 = model_train.for_input(real_sim.train.data)

        model_valid = model_train.for_input(real_sim.validation.data)

        with tf.Session().as_default():
            tf.global_variables_initializer().run()
            print(np.sum(model_train.Ws[0].eval() - model_valid.Ws[0].eval()))
            print(np.sum(model_train.bs[0].eval() - model_valid.bs[0].eval()))

            print(np.sum(model_train.inp[-1].eval() - model_train2.inp[-1].eval()))
            print(np.sum(model_train.inp[-1].eval() - model_train2.inp[-1].eval()))
            print(np.sum(model_train.inp[-1].eval() - model_train2.inp[-1].eval()))

            print(np.sum(model_train.inp[-1].eval() - model_train.inp[-1].eval()))
            print(np.sum(model_train.inp[-1].eval() - model_train.inp[-1].eval()))
            # if sparse matmul is used then these last values are not 0!


if __name__ == '__main__':
    # unittest.main()
    TestModels().test_sparse_input_models()
