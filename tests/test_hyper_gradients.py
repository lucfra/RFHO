import unittest
import numpy as np
from tests.test_base import iris_logistic_regression
from rfho.datasets import load_iris, load_mnist
from rfho.models import *
from rfho.hyper_gradients import ReverseHG, ForwardHG, HyperOptimizer
from rfho.optimizers import *
from rfho.utils import dot, SummaryUtil, SummaryUtils as SSU, PrintUtils, norm, stepwise_pu, MergedUtils, \
    cross_entropy_loss
import tabulate

def build_model(augment=0, variable_initializer=(tf.zeros, tf.zeros)):
    """
    Simple model for test purposes on minist
    
    :param augment: 
    :param variable_initializer: 
    :return: 
    """
    mnist = load_mnist()
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    lin_model = LinearModel(x, 28 * 28, 10,
                            active_gen_kwars={'init_w': variable_initializer[0],
                                              'init_b': variable_initializer[1]})

    all_w, mod_y, mat_w = vectorize_model(lin_model.var_list, lin_model.inp[-1], lin_model.Ws[0], augment=augment)

    error = tf.reduce_mean(
        cross_entropy_loss(mod_y, y)
    )

    correct_prediction = tf.equal(tf.argmax(mod_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return mnist, x, y, all_w, mod_y, mat_w, error, accuracy


class TestD(unittest.TestCase):

    def _test_doh_iris(self):  # FIXME to check. Probably now does not work

        tf.reset_default_graph()

        for _ in range(1):
            with tf.Graph().as_default():
                iris = load_iris()
                x = tf.placeholder(tf.float32, name='x')
                y = tf.placeholder(tf.float32, name='y')
                # net = FFNN(x, [4, 20, 20, 20, 3])
                net = LinearModel(x, 4, 3)
                net_w, net_out = vectorize_model(net.var_list, net.inp[-1])

                l2_factor = tf.Variable(.001, name='l2_factor')
                eta = tf.Variable(.1, name='learning_rate')

                error = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(net_out, y)
                )
                tr_error = error + l2_factor * dot(net_w, net_w)

                hyper_list = [l2_factor, eta]
                doh = ReverseHG(GradientDescentOptimizer.create(
                    net_w, eta, loss=tr_error), hyper_list, error, [])

                T = 2000

                def training_supplier(step): return {x: iris.train.data, y: iris.train.target}

                def validation_supplier(step=None): return {x: iris.test.data, y: iris.test.target}

                with tf.name_scope('summaries'):  # write summaries here?
                    s_err = tf.summary.scalar('error', error)
                    s_np = tf.summary.scalar('squared_norm_p', dot(doh.p_dict, doh.p_dict))
                    s_hyper_der = [tf.summary.scalar('hyper/'+hy.name, hy_der)
                                   for hy, hy_der in zip(hyper_list, doh.hyper_derivatives)]

                fw_merged = tf.summary.merge([s_err])
                bw_merged = tf.summary.merge([s_np] + s_hyper_der)

                su = SummaryUtil(
                    ops=fw_merged, writer='summary_test/fw',
                    condition=lambda step: step % 10 == 0, fd_supplier=training_supplier
                )

                su_bk = SummaryUtil(
                    ops=bw_merged, writer='summary_test/bw',
                    condition=lambda step: step % 10 == 0, fd_supplier=training_supplier
                )

                pn = norm(doh.p_dict)

                pu_bk = PrintUtils(stepwise_pu(lambda ss, step: ss.run(pn), 100))

                bk_merged = MergedUtils(pu_bk, SSU(su_bk))

                with tf.Session().as_default():
                    tf.variables_initializer(hyper_list).run()

                    hyper_grads = doh.run_all(
                        T, training_supplier, validation_supplier,
                        forward_su=SSU(su), backward_su=bk_merged
                    )

                print(hyper_grads)
                # print(sum(sum(hyper_grads, [])))

    def _test_doh_mnist(self):  # FIXME to check. Probably now does not work

        tf.reset_default_graph()

        for _ in range(1):
            with tf.Graph().as_default():
                mnist = load_mnist()
                x = tf.placeholder(tf.float32, name='x')
                y = tf.placeholder(tf.float32, name='y')
                # net = FFNN(x, [4, 20, 20, 20, 3])
                net = LinearModel(x, 28*28, 10)
                net_w, net_out = vectorize_model(net.var_list, net.inp[-1])

                # l2_factor = tf.Variable(.001, name='l2_factor')
                # gamma = tf.Variable(tf.random_uniform([28*28*10 + 10], minval=0., maxval=.001), name='reg_factors')
                    # tf.zeros([28 * 28*10 + 10 ]), name='regularizer_weights')
                # regularizer = tf.reduce_sum(gamma * net_w ** 2)

                eta = tf.Variable(.1, name='learning_rate')

                error = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(net_out, y)
                )

                net_true_out = tf.nn.softmax(net_out)

                error = tf.reduce_mean(- tf.reduce_sum(
                    y*tf.log(net_true_out), reduction_indices=[1])) # this is the true error... that works.. the other
                # made with tf.nn.softmax_cross_entropy does not!!!!!!!

                # tr_error = error +  * dot(net_w, net_w)
                tr_error = error  # + regularizer

                # hyper_list = [l2_factor, eta]
                hyper_list = [eta]  # , gamma]
                doh = ReverseHG(GradientDescentOptimizer.create(
                    net_w, eta, loss=tr_error), hyper_list, error, [])

                pn = norm(doh.p_dict)  # monitor for the norm of costate

                T = 1000

                def training_supplier(step=None):
                    bx, by = mnist.train.next_batch(128)
                    return {x: bx, y: by}

                def validation_supplier(step=None):
                    return {x: mnist.test.images, y: mnist.test.labels}

                with tf.name_scope('summaries'):  # write summaries here?
                    s_err = tf.summary.scalar('error', error)
                    s_np = tf.summary.scalar('squared_norm_p', pn)
                    # s_hyper_der = [tf.summary.scalar('hyper/'+hy.name, hy_der)
                    #               for hy, hy_der in zip(hyper_list, doh.hyper_derivatives)]

                fw_merged = tf.summary.merge([s_err])
                bw_merged = tf.summary.merge([s_np]) # + s_hyper_der)

                su = SummaryUtil(
                    ops=fw_merged, writer='summary_test/fw',
                    condition=lambda step: step % 10 == 0, fd_supplier=training_supplier
                )

                su_bk = SummaryUtil(
                    ops=bw_merged, writer='summary_test/bw',
                    condition=lambda step: step % 10 == 0, fd_supplier=training_supplier
                )

                pu_bk = PrintUtils(stepwise_pu(lambda ss, step: ss.run(pn), 100))

                bk_merged = MergedUtils(pu_bk, SSU(su_bk))

                with tf.Session().as_default():
                    tf.variables_initializer(hyper_list).run()

                    hyper_grads = doh.run_all(
                        T, training_supplier, validation_supplier,
                        forward_su=SSU(su), backward_su=bk_merged
                    )

                print(hyper_grads)
                # print(sum(sum(hyper_grads, [])))

    def _test_doh_gd(self):  # FIXME to check. Probably now does not work
        tf.reset_default_graph()

        mnist, x, y, all_w, mod_y, mod_w, error, accuracy = build_model()
        # till here must be included!

        ts = tf.train.GradientDescentOptimizer(.1).minimize(error, var_list=[all_w])

        bxs, bys = ([], [])

        psu = None
        with tf.Session().as_default() as ss:
            tf.variables_initializer([all_w]).run()

            for _ in range(1000):
                bx, by = mnist.train.next_batch(128)
                bxs.append(bx)  #
                bys.append(by)

                ss.run(ts, feed_dict={x: bx, y: by})
                # if _ % 100 == 0:
                #     print(ss.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
                if psu:
                    psu.run(ss, _)

            final_w = all_w.eval()

        def training_supplier(step): return {x: bxs[step], y: bys[step]}

        doh = ReverseHG(GradientDescentOptimizer.create(
            all_w, .1, loss=error), [], error, [])

        with tf.Session().as_default():
            doh.forward(1000, training_supplier, summary_utils=psu)

            final_w_doh = all_w.eval()

        self.assertLess(np.linalg.norm(final_w - final_w_doh), 1e-5)

    def test_adam_standalone(self):
        """
        Makes sure that the adam implemented in doh is almost equal to the one of tensorflow.
        There is a little numeric difference due to the different way to calculate
        the bias correction term (tensorflow accumulates the powers of moment factors into variables, but that
        operation becomes not differentiable, so the moment factors beta_1 and beta_2 cannot be optimized).
        :return:
        """
        tf.reset_default_graph()

        v = tf.Variable([1., 2., 3.])
        obj = tf.reduce_sum(tf.pow(v, 2))

        iterations = 100
        lr = .1

        adam_dict = AdamOptimizer.create(
            v, lr=lr, loss=obj, w_is_state=False)
        print(adam_dict)

        print(tf.global_variables())

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                print(adam_dict.dynamics.eval())
                ss.run(adam_dict.assign_ops)
                adam_dict.increase_global_step()
            res = v.eval()

        print(res)

        tf_adam = tf.train.AdamOptimizer(learning_rate=lr).minimize(obj, var_list=[v])

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                ss.run(tf_adam)
            res2 = v.eval()

        print(res2)

        self.assertLess(np.linalg.norm(res - res2), 1.e-5)

    def test_adam_standalone_with_augmentation(self):
        """
        Makes sure that the adam implemented in doh is almost equal to the one of tensorflow.
        There is a little numeric difference due to the different way to calculate
        the bias correction term (tensorflow accumulates the powers of moment factors into variables, but that
        operation becomes not differentiable, so the moment factors beta_1 and beta_2 cannot be optimized).
        :return:
        """
        tf.reset_default_graph()

        v = tf.Variable([1., 2., 3.])
        obj = tf.reduce_sum(tf.pow(v, 2))

        v1, obj1 = vectorize_model([v], obj, augment=2)

        iterations = 100
        lr = .1

        adam_dict = AdamOptimizer.create(v1, lr=lr, loss=obj1, w_is_state=True)
        # assign_ops = v1.assign(adam_dict)

        # print(assign_ops)

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                print(ss.run(adam_dict.dynamics))
                ss.run(adam_dict.assign_ops)
                adam_dict.increase_global_step()
            res = v.eval()
            print(v1.tensor.eval())

        print(res)

        tf_adam = tf.train.AdamOptimizer(learning_rate=lr).minimize(obj, var_list=[v])

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                ss.run(tf_adam)
            res2 = v.eval()

        print(res2)

        self.assertLess(np.linalg.norm(res - res2), 1.e-5)

    def test_momentum_standalone(self):
        """
        Makes sure that the adam implemented in doh is almost equal to the one of tensorflow.
        There is a little numeric difference due to the different way to calculate
        the bias correction term (tensorflow accumulates the powers of moment factors into variables, but that
        operation becomes not differentiable, so the moment factors beta_1 and beta_2 cannot be optimized).
        :return:
        """
        tf.reset_default_graph()

        v = tf.Variable([1., 2., 3.])
        obj = tf.reduce_sum(tf.pow(v, 2))

        iterations = 5
        lr = .5
        mu = .5

        momentum_dict = MomentumOptimizer.create(v, lr=lr, mu=mu, loss=obj, w_is_state=False)
        print(momentum_dict)

        print(tf.global_variables())

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                print(momentum_dict.dynamics.eval())
                ss.run(momentum_dict.assign_ops)
            res = v.eval()

        print(res)

        mom_opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=.5)

        ts_momentum = mom_opt.minimize(obj, var_list=[v])

        print(mom_opt.get_slot_names())

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                print(v.eval(), mom_opt.get_slot(v, 'momentum').eval())
                ss.run(ts_momentum)
            res2 = v.eval()

        print(res2)

        self.assertLess(np.linalg.norm(res - res2), 1.e-5)

    def test_momentum_with_augmentation(self):
        """
        Makes sure that the adam implemented in doh is almost equal to the one of tensorflow.
        There is a little numeric difference due to the different way to calculate
        the bias correction term (tensorflow accumulates the powers of moment factors into variables, but that
        operation becomes not differentiable, so the moment factors beta_1 and beta_2 cannot be optimized).
        :return:
        """
        tf.reset_default_graph()

        v = tf.Variable([1., 2., 3.])
        obj = tf.reduce_sum(tf.pow(v, 2))
        v1, obj1 = vectorize_model([v], obj, augment=1)

        iterations = 5
        lr = .5
        mu = .5

        momentum_dict = MomentumOptimizer.create(v1, lr=lr, mu=mu, loss=obj1, w_is_state=True)
        print(momentum_dict)

        print(tf.global_variables())

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                print(momentum_dict.dynamics.eval())
                ss.run(momentum_dict.assign_ops)
            res = v.eval()

        print(res)

        mom_opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=.5)

        ts_momentum = mom_opt.minimize(obj, var_list=[v])

        print(mom_opt.get_slot_names())

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for _ in range(iterations):
                print(v.eval(), mom_opt.get_slot(v, 'momentum').eval())
                ss.run(ts_momentum)
            res2 = v.eval()

        print(res2)

        self.assertLess(np.linalg.norm(res - res2), 1.e-5)

    def test_hyper_gradient_none_check(self):
        _w = tf.Variable(10.)
        w, ww = vectorize_model([_w], _w)
        a, b, c = tf.Variable(10.), tf.Variable(10.), tf.Variable(10.)
        f = ww**2 + ww*a + b
        optimizer = GradientDescentOptimizer.create(w, .1, loss=f)
        with self.assertRaises(AssertionError):
            # all this configurations should rise error!
            hyper_dict = {f: [a, b, c]}
            ReverseHG(optimizer, hyper_dict)
            ReverseHG(optimizer, {f: c})
            ReverseHG(optimizer, {f: b})
        self.assertTrue(ReverseHG(optimizer, {f: a}))  # while this one should be fine

    def _bkfw_test(self, param_optimizer, method, debug_jac=False, iterations=100):
        tf.set_random_seed(0)
        np.random.seed(0)
        iris, x, y, model, model_w, model_y, error, accuracy = iris_logistic_regression(
            param_optimizer.get_augmentation_multiplier())

        # rho = tf.Variable([.1, .01], name='rho')
        tr_error = error \
                   # + rho[0]*tf.reduce_sum(model_w.tensor**2)\
                   # + rho[1]*tf.abs(tf.reduce_sum(model_w.tensor))

        eta = tf.Variable(.001, name='eta')
        dyn = param_optimizer.create(model_w, eta, loss=tr_error, _debug_jac_z=debug_jac)
        tr_sup = lambda s=None: {x: iris.train.data, y: iris.train.target}
        val_sup = lambda s=None: {x: iris.validation.data, y: iris.validation.target}

        hy_opt = HyperOptimizer(dyn, {error: [eta
                                              ]
            # , rho]
        }, method=method)
        all_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        with tf.Session().as_default() as ss:
            hy_opt.initialize()
            for k in range(1):
                hy_opt.run(iterations, tr_sup, {error: val_sup}, _debug_no_hyper_update=True)
                # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
                if 'direct_HO/Jacobian' in all_names:
                    jac = ss.run('direct_HO/Jacobian:0', feed_dict=tr_sup())
                    np.savetxt('temp.csv', jac, delimiter=',')
                if 'direct_HO/Jacobian_cal' in all_names:
                    jac = ss.run('direct_HO/Jacobian_cal:0', feed_dict=tr_sup())
                    np.savetxt('temp_cal.csv', jac, delimiter=',')
            hyp_gs = ss.run(hy_opt.hyper_gradients.hyper_gradient_vars)
            print(hyp_gs)

            print()
            norm_of_state = np.linalg.norm(model_w.eval())
            print('norm of state vector', norm_of_state)  # this check is passed...
            print()
            return hyp_gs, norm_of_state


    def trial_solve0(self):
        # unittest.main()
        test = TestD()
        n_iters = 100
        trials = 10
        res_f = []
        res_r = []
        for j in range(trials):
            test.setUp()
            hgf, nof = test._bkfw_test(param_optimizer=AdamOptimizer,
                                       method=ForwardHG, debug_jac=False, iterations=n_iters)
            # TestD()._bkfw_test(method=ForwardHG, debug_jac=False)

            test.setUp()
            hgr, nor = test._bkfw_test(param_optimizer=AdamOptimizer,
                                       method=ReverseHG, iterations=n_iters)

            print(nof - nor)
            res_f.append(hgf)
            res_r.append(hgr)

        for hgf, hgr in zip(res_f, res_r):
            # TestD().test_momentum(method=ForwardHG)
            print([(hf - hr) for hr, hf in zip(hgr, hgf)])
            # tf.reset_default_graph()
            # TestD().test_momentum(method=ReverseHG)

        print()
        print(res_f)
        print()
        print(res_r)


    def _test_hv(self, param_optimizer, debug_jac=False, iterations=100):
        tf.set_random_seed(0)
        np.random.seed(0)
        iris, x, y, model, model_w, model_y, error, accuracy = iris_logistic_regression(
            param_optimizer.get_augmentation_multiplier())

        eta = tf.Variable(.001, name='eta')
        dyn = param_optimizer.create(model_w, eta, loss=error, _debug_jac_z=debug_jac)

        # # rho = tf.Variable([.1, .01], name='rho')
        # tr_error = error
        #            # + rho[0]*tf.reduce_sum(model_w.tensor**2)\
        #            # + rho[1]*tf.abs(tf.reduce_sum(model_w.tensor))

        tr_sup = lambda s=None: {x: iris.train.data, y: iris.train.target}

        from rfho.utils import hvp

        z = tf.ones(model_w.get_shape())
        hv = hvp(error, model_w.tensor, z)

        with tf.Session().as_default() as ss:
            tf.global_variables_initializer().run()
            for t in range(iterations):
                ss.run(dyn.assign_ops, feed_dict=tr_sup())
            return hv.eval(feed_dict=tr_sup())

        # with tf.Session().as_default() as ss:

    def do_test_hv(self):
        trials = 100
        rs = []
        for tr in range(trials):
            self.setUp()
            rs.append(self._test_hv(AdamOptimizer))
        for r in rs:
            print(r)

        self.assertEqual(len(set(rs)), 1)



    def _forward_fix_test(self, param_optimizer, method, debug_jac=False, iterations=100):
        tf.set_random_seed(0)
        np.random.seed(0)
        iris, x, y, model, model_w, model_y, error, accuracy = iris_logistic_regression(
            param_optimizer.get_augmentation_multiplier())

        # rho = tf.Variable([.1, .01], name='rho')
        tr_error = error \
                   # + rho[0]*tf.reduce_sum(model_w.tensor**2)\
                   # + rho[1]*tf.abs(tf.reduce_sum(model_w.tensor))

        eta = tf.Variable(.001, name='eta')
        dyn = param_optimizer.create(model_w, eta, loss=tr_error, _debug_jac_z=debug_jac)
        tr_sup = lambda s=None: {x: iris.train.data, y: iris.train.target}
        val_sup = lambda s=None: {x: iris.validation.data, y: iris.validation.target}

        hy_opt = HyperOptimizer(dyn, {error: [eta
                                              ]
            # , rho]
        }, method=method)
        zs, gves = [], []
        all_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        with tf.Session().as_default() as ss:
            hy_opt.initialize()
            for k in range(1):
                for it in range(iterations):
                    zs.append(ss.run(hy_opt.hyper_gradients.zs[0].tensor))
                    hy_opt.run(1, tr_sup, {error: val_sup}, _debug_no_hyper_update=True)
                zs.append(ss.run(hy_opt.hyper_gradients.zs[0].tensor))
                gves.append(ss.run(hy_opt.hyper_gradients.grad_val_err, feed_dict=val_sup()))
                # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
            hyp_gs = ss.run(hy_opt.hyper_gradients.hyper_gradient_vars)
            print(hyp_gs)

            print()
            norm_of_state = np.linalg.norm(model_w.eval())
            print('norm of state vector', norm_of_state)  # this check is passed...
            print()
            return hyp_gs, zs, gves

    def setUp(self):
        tf.reset_default_graph()


def trial_solve1():
    test = TestD()
    n_iters = 100
    trials = 200
    res_f = []
    res_z = []
    res_gve = []
    for j in range(trials):
        test.setUp()
        hgf, zs, gve = test._forward_fix_test(param_optimizer=AdamOptimizer,
                                              method=ForwardHG, debug_jac=False, iterations=n_iters)
        res_gve.append(gve)
        res_f.append(hgf)
        res_z.append(zs)

    tbr = []
    for rs, z, gve in zip(res_f, res_z, res_gve):
        _ = np.linalg.norm(np.array(z))
        norm2 = np.linalg.norm(gve)
        tbr.append((rs[0], _, norm2))
    print(tabulate.tabulate(tbr))

if __name__ == '__main__':
    # trial_solve1()
    TestD().do_test_hv()