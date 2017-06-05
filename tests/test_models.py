import unittest
from tests.test_base import *
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

    # FIXME (last time error:tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.
    # def test_sparse_input_models(self):
    #     import rfho.datasets as ddt
    #
    #     real_sim = ddt.load_20newsgroup_vectorized(partitions_proportions=[.5, .3])
    #
    #     model_train = LinearModel(real_sim.train.data, real_sim.train.dim_data, real_sim.train.dim_target,
    #                                init_w=tf.random_normal, init_b=tf.random_normal, benchmark=False)
    #
    #     model_train2 = model_train.for_input(real_sim.train.data)
    #
    #     model_valid = model_train.for_input(real_sim.validation.data)
    #
    #     with tf.Session().as_default():
    #         tf.global_variables_initializer().run()
    #         self.assertEqual(np.sum(model_train.Ws[0].eval()),  np.sum(model_valid.Ws[0].eval()))
    #         self.assertEqual(np.sum(model_train.bs[0].eval()), np.sum(model_valid.bs[0].eval()))
    #
    #         print(np.sum(model_train.inp[-1].eval() - model_train2.inp[-1].eval()))
    #         print(np.sum(model_train.inp[-1].eval() - model_train2.inp[-1].eval()))
    #         print(np.sum(model_train.inp[-1].eval() - model_train2.inp[-1].eval()))
    #
    #         print(np.sum(model_train.inp[-1].eval() - model_train.inp[-1].eval()))
    #         print(np.sum(model_train.inp[-1].eval() - model_train.inp[-1].eval()))
    #         # if sparse matmul is used then on gpu these last values are not 0!

    def test_ffnn(self):
        x, y = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
        model = FFNN(x, dims=[10, 123, 89, 47], active_gen_kwargs=(
            {'activ': mixed_activation(tf.identity, tf.nn.sigmoid)},
            {}
        ))

        mod_y = model.for_input(y)
        self.assertEqual(model.var_list, mod_y.var_list)  # share variables
        self.assertNotEqual(model.inp[1:], mod_y.inp[1:])  # various activations are different nodes!

    def test_simpleCNN(self):
        x, y = tf.placeholder(tf.float32), tf.placeholder(tf.float32)
        model = SimpleCNN(tf.reshape(x, [-1, 15, 15, 1]),
                     conv_dims=[
                         [5, 5, 1, 4],
                         [5, 5, 4, 8],
                     ], ffnn_dims=[20, 10],
                          active_gen_kwargs=(
                              {'activ': mixed_activation(tf.identity, tf.nn.sigmoid)},
                              {}
                          ))

        mod_y = model.for_input(tf.reshape(y, [-1, 15, 15, 1]))

        self.assertEqual(model.ffnn_part.var_list, mod_y.ffnn_part.var_list)

        self.assertEqual(model.var_list, mod_y.var_list)  # share variables
        self.assertNotEqual(model.inp[1:], mod_y.inp[1:])  # various activations are different nodes!
        w, out, out_y = vectorize_model(model.var_list, model.inp[-1], mod_y.inp[-1])
        self.assertIsNotNone(out)

    def test_determinitstic_initalization(self):
        x = tf.constant([[1., 2.]])
        mod = FFNN(x, [2, 2, 1], deterministic_initialization=True)

        with tf.Session().as_default() as ss:
            mod.initialize()

            vals = ss.run(mod.Ws)

            mod.initialize()

            assert_array_lists_same(ss.run(mod.Ws), vals, test_case=self)

        print()

        with tf.Session().as_default() as ss:
            mod.initialize()

            assert_array_lists_same(ss.run(mod.Ws), vals, test_case=self)


if __name__ == '__main__':
    unittest.main()
    # TestModels().test_determinitstic_initalization()
