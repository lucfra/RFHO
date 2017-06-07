import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from rfho.models import LinearModel, vectorize_model
from rfho.utils import cross_entropy_loss, stepwise_pu, unconditional_pu, PrintUtils, norm
from rfho.datasets import load_iris, ExampleVisiting
from rfho.hyper_gradients import ReverseHG, ForwardHG
from rfho.optimizers import *
import unittest


class TestDohDirectDoh(unittest.TestCase):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def test_single_hp(self):
        tf.reset_default_graph()

        T = 100
        lr = .01

        hyper_iterations = 10

        hyper_learning_rate = .001

        iris = load_iris([.4, .4])
        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        model = LinearModel(x, 4, 3)
        w, model_out = vectorize_model(model.var_list, model.inp[-1])

        error = tf.reduce_mean(cross_entropy_loss(model_out, y))

        correct_prediction = tf.equal(tf.argmax(model_out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        eta = tf.Variable(lr, name='eta')
        dynamics_dict = GradientDescentOptimizer.create(w, lr=eta, loss=error)

        doh = ReverseHG(dynamics_dict, hyper_dict={error: eta})

        grad = tf.gradients(error, w.tensor)[0]

        hyper_dict = {error: (eta, -grad)}

        direct_doh = ForwardHG(dynamics_dict, hyper_dict=hyper_dict)

        # noinspection PyUnusedLocal
        def all_training_supplier(step=None): return {x: iris.train.data, y: iris.train.target}

        training_supplier = all_training_supplier

        # noinspection PyUnusedLocal
        def validation_supplier(step=None): return {x: iris.validation.data, y: iris.validation.target}

        # noinspection PyUnusedLocal
        def test_supplier(step=None): return {x: iris.test.data, y: iris.test.target}

        psu = PrintUtils(
            stepwise_pu(lambda ses, step: print('test accuracy', ses.run(accuracy, feed_dict=test_supplier())), T - 1),
        )
        psu2 = None

        history_test_accuracy = []
        history_eta = []

        # noinspection PyUnusedLocal
        def save_accuracies(ses, step):
            history_test_accuracy.append(ses.run(accuracy, feed_dict=test_supplier()))
            history_eta.append(ses.run(eta))

        after_forward_su = PrintUtils(unconditional_pu(save_accuracies), unconditional_pu(
            lambda ses, step: print('training error', error.eval(feed_dict=all_training_supplier()))))

        delta_hyper = tf.placeholder(tf.float32)

        hyper_upd_ops = [hyp.assign(tf.minimum(tf.maximum(hyp - delta_hyper, tf.zeros_like(hyp)), tf.ones_like(hyp)))
                         for hyp in doh.hyper_list]  # check the sign of gradient

        # In[ ]:
        diffs = []
        with tf.Session(config=TestDohDirectDoh.config).as_default() as ss:
            tf.variables_initializer([eta]).run()

            for _ in range(hyper_iterations):

                direct_doh.initialize()
                for _k in range(T):
                    direct_doh.step_forward(train_feed_dict_supplier=training_supplier, summary_utils=psu)

                direct_res = direct_doh.hyper_gradient_vars(validation_suppliers=training_supplier)

                res = doh.run_all(T, train_feed_dict_supplier=training_supplier, after_forward_su=after_forward_su,
                                  val_feed_dict_suppliers=training_supplier, forward_su=psu, backward_su=psu2)

                collected_hyper_gradients = list(ReverseHG.std_collect_hyper_gradients(res).values())
                [ss.run(hyper_upd_ops[j],
                        feed_dict={delta_hyper: hyper_learning_rate * collected_hyper_gradients[j]})
                 for j in range(len(doh.hyper_list))]

                self.assertLess(np.linalg.norm(np.array(direct_res[eta]) - np.array(collected_hyper_gradients)),
                                1.e-5)

                diffs.append(np.array(direct_res[eta]) - np.array(collected_hyper_gradients))

        ev_data = ExampleVisiting(iris, 10, 10)
        T = ev_data.T

        training_supplier = ev_data.create_train_feed_dict_supplier(x, y)

        with tf.Session(config=TestDohDirectDoh.config).as_default() as ss:
            tf.variables_initializer([eta]).run()

            for _ in range(hyper_iterations):

                ev_data.generate_visiting_scheme()

                direct_doh.initialize()

                for _k in range(T):
                    direct_doh.step_forward(train_feed_dict_supplier=training_supplier, summary_utils=psu)

                direct_res = direct_doh.hyper_gradient_vars(validation_suppliers=all_training_supplier)

                res = doh.run_all(T, train_feed_dict_supplier=training_supplier, after_forward_su=after_forward_su,
                                  val_feed_dict_suppliers=all_training_supplier, forward_su=psu, backward_su=psu2)

                collected_hyper_gradients = list(ReverseHG.std_collect_hyper_gradients(res).values())
                [ss.run(hyper_upd_ops[j],
                        feed_dict={delta_hyper: hyper_learning_rate * collected_hyper_gradients[j]})
                 for j in range(len(doh.hyper_list))]

                self.assertLess(np.linalg.norm(np.array(direct_res[eta]) - np.array(collected_hyper_gradients)),
                                1.e-5)

    # def _test_multiple_hp(self, momentum=False):  # FIXME update this test
    #     tf.reset_default_graph()
    #
    #     T = 100
    #     lr = .01
    #
    #     hyper_iterations = 10
    #
    #     hyper_learning_rate = .001
    #
    #     mu = None
    #     if momentum:
    #         mu = tf.Variable(.7, name='mu')
    #
    #     iris = load_iris([.4, .4])
    #
    #     x = tf.placeholder(tf.float32, name='x')
    #     y = tf.placeholder(tf.float32, name='y')
    #     model = LinearModel(x, 4, 3)
    #     w, model_out, mat_W, b = vectorize_model(model.var_list, model.inp[-1], model.Ws[0], model.bs[0],
    #                                              augment=momentum)
    #
    #     error = tf.reduce_mean(cross_entropy_loss(model_out, y))
    #
    #     gamma = tf.Variable([0., 0.], name='gamma')
    #     regularizer = gamma[0]*tf.reduce_sum(mat_W**2) + gamma[1]*tf.reduce_sum(b**2)
    #
    #     training_error = error + regularizer
    #
    #     correct_prediction = tf.equal(tf.argmax(model_out, 1), tf.argmax(y, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #
    #     eta = tf.Variable(lr, name='eta')
    #     if momentum:
    #         dynamics_dict = MomentumOptimizer.create(w, lr=eta, mu=mu, loss=training_error)
    #     else:
    #         dynamics_dict = GradientDescentOptimizer.create(w, lr=eta, loss=training_error)
    #
    #     if momentum:
    #         doh = ReverseHG(dynamics_dict, hyper_dict={training_error: [eta, mu], error: [gamma]})
    #     else:
    #         doh = ReverseHG(dynamics_dict, hyper_dict={training_error: [eta], error: [gamma]})
    #
    #     # In[8]:
    #     true_w = w.var_list(VlMode.TENSOR)[0]
    #     grad = tf.gradients(training_error, true_w)[0]
    #
    #     _grad_reg = tf.gradients(regularizer, gamma)[0]
    #     grad_reg = tf.stack([
    #         tf.gradients(_grad_reg[0], true_w)[0], tf.gradients(_grad_reg[1], true_w)[0]
    #     ], axis=1)
    #
    #     if momentum:
    #         w_b, m = w.var_list(VlMode.TENSOR)
    #         # noinspection PyUnresolvedReferences
    #         grad = ZMergedMatrix([
    #             - tf.transpose([mu * m + grad]),
    #             tf.zeros([m.get_shape().as_list()[0], 1])
    #         ])
    #         grad_reg = ZMergedMatrix([
    #             -eta*grad_reg,
    #             grad_reg
    #         ])
    #         grad_mu = ZMergedMatrix([
    #             - (eta * m),
    #             m
    #         ])
    #
    #     else:
    #         grad_mu = None
    #         grad_reg *= eta
    #
    #     if momentum:
    #         hyper_dict = {training_error: [(eta, grad), (mu, grad_mu)],
    #                       error: (gamma, grad_reg)}
    #         direct_doh = ForwardHG(dynamics_dict, hyper_dict=hyper_dict)
    #     else:
    #         hyper_dict = {training_error: (eta, -grad),
    #                       error: (gamma, -grad_reg)}
    #         direct_doh = ForwardHG(dynamics_dict, hyper_dict=hyper_dict)
    #
    #     # noinspection PyUnusedLocal
    #     def all_training_supplier(step=None): return {x: iris.train.data, y: iris.train.target}
    #
    #     training_supplier = all_training_supplier
    #
    #     # noinspection PyUnusedLocal
    #     def validation_supplier(step=None): return {x: iris.validation.data, y: iris.validation.target}
    #
    #     # noinspection PyUnusedLocal
    #     def test_supplier(step=None): return {x: iris.test.data, y: iris.test.target}
    #
    #     psu = PrintUtils(
    #         stepwise_pu(lambda ses, step: print('test accuracy', ses.run(accuracy, feed_dict=test_supplier())), T - 1),
    #     )
    #     norm_p = norm(tf.concat(list(doh.p_dict.values()), 0))
    #     psu2 = PrintUtils(stepwise_pu(
    #         lambda ses, step: print('norm of costate', ses.run(norm_p)), T - 1))
    #
    #     history_test_accuracy = []
    #     history_eta = []
    #
    #     # noinspection PyUnusedLocal
    #     def save_accuracies(ses, step):
    #         history_test_accuracy.append(ses.run(accuracy, feed_dict=test_supplier()))
    #         history_eta.append(ses.run(eta))
    #
    #     after_forward_su = PrintUtils(unconditional_pu(save_accuracies), unconditional_pu(
    #         lambda ses, step: print('training error', error.eval(feed_dict=all_training_supplier()))))
    #
    #     delta_hyper = tf.placeholder(tf.float32)
    #
    #     hyper_upd_ops = {hyp: hyp.assign(tf.maximum(hyp - delta_hyper, tf.zeros_like(hyp)))
    #                      for hyp in doh.hyper_list}  # check the sign of gradient
    #
    #     with tf.Session(config=TestDohDirectDoh.config).as_default() as ss:
    #         tf.variables_initializer(doh.hyper_list).run()
    #
    #         for _ in range(hyper_iterations):
    #
    #             direct_doh.initialize()
    #             for _k in range(T):
    #                 direct_doh.step_forward(train_feed_dict_supplier=training_supplier, summary_utils=psu)
    #
    #             validation_suppliers = {training_error: training_supplier, error: validation_supplier}
    #
    #             if momentum:
    #                 direct_res = direct_doh.hyper_gradient_vars(validation_suppliers=validation_suppliers)
    #             else:
    #                 direct_res = direct_doh.hyper_gradient_vars(validation_suppliers=validation_suppliers)
    #
    #             res = doh.run_all(T, train_feed_dict_supplier=training_supplier, after_forward_su=after_forward_su,
    #                               val_feed_dict_suppliers={error: validation_supplier, training_error: training_supplier},
    #                               forward_su=psu, backward_su=psu2)
    #
    #             collected_hyper_gradients = ReverseHG.std_collect_hyper_gradients(res)
    #
    #             [ss.run(hyper_upd_ops[hyp],
    #                     feed_dict={delta_hyper: hyper_learning_rate * collected_hyper_gradients[hyp]})
    #              for hyp in doh.hyper_list]
    #
    #             for hyp in doh.hyper_list:
    #                 self.assertLess(np.linalg.norm(
    #                     np.array(direct_res[hyp]) - np.array(collected_hyper_gradients[hyp])), 1.e-5)
    #
    # def test_multiple_hypers(self):
    #     self._test_multiple_hp(momentum=False)
    #     self._test_multiple_hp(momentum=True)

    def setUp(self):
        tf.reset_default_graph()

if __name__ == '__main__':
    # TestDohDirectDoh()._test_multiple_hp(True)
    unittest.main()
