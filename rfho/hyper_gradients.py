import numpy as np
import tensorflow as tf

from rfho.optimizers import Optimizer
from rfho.utils import dot, MergedVariable, Vl_Mode, as_list, simple_name, GlobalStep, ZMergedMatrix


class ReverseHyperGradient:

    # noinspection SpellCheckingInspection
    def __init__(self, optimizer, hyper_dict, state_history=None, global_step=None):
        """
        Creates a new object that computes the hyper-gradient of validation errors in reverse mode.
        See section 3.1 of Forward and Reverse Gradient-Based Hyperparameter Optimization
        (https://arxiv.org/abs/1703.01785)
        Note that this class only computes the hyper-gradient and does not perform hyperparameter optimization.

        :param optimizer: insance of Optimizer class, which contains the dynamics with which the model parameters are
                            updated
        :param hyper_dict: A dictionary of `{validation_error: hyperparameter or list_of_hyperparameters}` where
                            `validation_error` is a scalar tensor and `list_of_hyperparameters` is a list
                            of tensorflow variables that represents the hyperparameters
        :param state_history: (default: empty list) state history manager:
                                should implement methods `clear`, `append`, `__getitem__`
        :param global_step: optional instance of GlobalStep class
        """
        assert isinstance(optimizer, Optimizer)

        self.w = optimizer.raw_w  # might be variable or MergedVariable
        #  TODO check if it works also with w as simple Variable
        self.w_t = MergedVariable.get_tensor(self.w)  # this is always a tensor

        self.tr_dynamics = optimizer.dynamics
        assert isinstance(hyper_dict, dict), '%s not allowed type. Should be a dict of ' \
                                             '(tf.Tensor, hyperparameters)' % hyper_dict
        self.val_error_dict = hyper_dict

        self.hyper_list = []
        for k, v in hyper_dict.items():
            self.hyper_list += as_list(v)
            self.val_error_dict[k] = as_list(v)  # be sure that are all lists

        self.w_hist = state_history or []

        with self.w_t.graph.as_default():
            # global step
            self.global_step = global_step or GlobalStep()

            self._fw_ops = optimizer.assign_ops  # TODO add here when hyper-parameters are sequence

            # backward assign ops
            with tf.name_scope('backward'):
                # equation (9)
                p_T = {ve: tf.gradients(ve, self.w_t)[0] for ve, hyp_list in self.val_error_dict.items()}  # deltaE(s_t)

                self.p_dict = {ve: tf.Variable(pt, name='p') for ve, pt in p_T.items()}

                # for nullity check
                self._abs_sum_p = tf.reduce_sum(tf.stack([tf.reduce_sum(tf.abs(p), name='l1_p')
                                                         for p in self.p_dict.values()]))

                # build Lagrangian function
                with tf.name_scope('lagrangian'):
                    self.lagrangians_dict = {ve: dot(p, self.tr_dynamics) for ve, p in self.p_dict.items()}

                # TODO read below
                '''
                In the following {if else} block there are two ways of computing the the dynamics of the update
                 of the Lagrangian multipliers. The procedures SHOULD produce the same result,
                however, for some strange reason, if w is indeed a state varibale that contains auxiliary components
                (e.g. velocity in Momentum algorithm, ...) there is a difference in the two methods and
                the right one is the first one. This is possibly due to the order in wich the derivatives are
                 taken by tensorflow, but furhter investigation is necessary.
                '''
                # detects if some auxiliary variables are used.
                if isinstance(self.w, MergedVariable) and \
                        any([isinstance(v, MergedVariable) for v in self.w.var_list(Vl_Mode.RAW)]):
                    state_components = self.w.var_list(Vl_Mode.TENSOR)

                    # equation (8)
                    self.p_dynamics = {ve: tf.concat(tf.gradients(lagrangian, state_components), 0)
                                       for ve, lagrangian in self.lagrangians_dict.items()}
                else:
                    # equation (8)
                    self.p_dynamics = {ve: tf.gradients(lagrangian, self.w_t)[0]
                                       for ve, lagrangian in self.lagrangians_dict.items()}  # equation (7)

                self._bk_ops = [self.p_dict[ve].assign(self.p_dynamics[ve])
                                for ve in self.val_error_dict]  # TODO add here when hp are sequ.

            with tf.name_scope('w_history_ops'):
                self._w_placeholder = tf.placeholder(self.w_t.dtype)

                self._back_hist_op = self.w.assign(self._w_placeholder)

            with tf.name_scope('hyper_derivatives'):
                # equation (10) without summation.
                self.hyper_derivatives = [
                    (self.val_error_dict[ve], tf.gradients(lagrangian, self.val_error_dict[ve])) for ve, lagrangian in
                    self.lagrangians_dict.items()
                    ]  # list of couples (hyper_list, list of symbolic hyper_gradients)  (lists are unhashable!)

            with tf.name_scope('hyper_gradients'):  # ADDED 28/3/17 keeps track of hyper-gradients as tf.Variable
                self._grad_wrt_hypers_placeholder = tf.placeholder(tf.float32, name='placeholder')
                # TODO this placeholder is not really necessary... just added to minimize the changes needed
                # (merge with RICCARDO)

                self.hyper_gradient_vars = [tf.Variable(tf.zeros_like(hyp), name=simple_name(hyp))
                                            for hyp in self.hyper_list]
                self.hyper_gradients_dict = {hyp: hgv for hyp, hgv  # redundant.. just for comfort ..
                                             in zip(self.hyper_list, self.hyper_gradient_vars)}

                self._hyper_assign_ops = {h: v.assign(self._grad_wrt_hypers_placeholder)
                                          for h, v in self.hyper_gradients_dict.items()}

    def initialize(self):
        """
        Helper for initializing all the variables. Builds and runs model variables and global step initializers.
        Note that dual variables are initialized only when calling `backward`.

        :return: None
        """
        assert tf.get_default_session() is not None, 'No default tensorflow session!'
        var_init = self.w.var_list(Vl_Mode.BASE) if isinstance(self.w, MergedVariable) else [self.w]
        tf.variables_initializer(var_init + self.hyper_gradient_vars + [self.global_step.var]).run()

    def forward(self, T, train_feed_dict_supplier=None, summary_utils=None):
        """
        Performs (forward) optimization of the parameters.

        :param T: Total number of iterations
        :param train_feed_dict_supplier: (optional) A callable with signature `t -> feed_dict` to pass to `tf.Session.run`
                                    feed_dict argument
        :param summary_utils: (optional) object that implements a method method `run(tf.Session, step)`
                                that is executed at every iteration (see for instance utils.PrintUtils
        :return: None
        """
        if not train_feed_dict_supplier:
            # noinspection PyUnusedLocal
            def feed_dict_supplier(step=None): return None

        # var_init = self.w.var_list(Vl_Mode.BASE) if isinstance(self.w, MergedVariable) else [self.w]
        # tf.variables_initializer(var_init + [self.global_step.var]).run()

        ss = tf.get_default_session()
        self.w_hist.clear()

        for t in range(T):
            self.w_hist.append(self.w_t.eval())
            ss.run([self.w_t, self._fw_ops, self.global_step.increase],
                   feed_dict=train_feed_dict_supplier(self.global_step.eval()))
            if summary_utils:
                summary_utils.run(ss, t)

    def backward(self, T, val_feed_dict_suppliers=None, train_feed_dict_supplier=None,
                 summary_utils=None, check_if_zero=False):
        """
        Performs backward computation of hyper-gradients

        :param T: Total number of iterations
        :param val_feed_dict_suppliers: either a callable that returns a feed_dict
                                        or a dictionary {validation_error tensor: callable (step -> feed_dict) that
                                        is used to initialize the dual variables `p` (generally supplier of
                                        validation example set).
        :param train_feed_dict_supplier: (optional) A callable with signature `t -> feed_dict` to pass to `tf.Session.run`
                                    feed_dict argument
        :param summary_utils: (optional) object that implements a method method `run(tf.Session, step)`
                                that is executed at every iteration (see for instance utils.PrintUtils
        :param check_if_zero: (optional) debug flag
        :return: A dictionary of lists of step-wise hyper-gradients. In usual application the "true" hyper-gradients
                 can be obtained with method std_collect_hyper_gradients
        """
        if not train_feed_dict_supplier:
            # noinspection PyUnusedLocal
            def training_supplier(step=None): return None
        if not val_feed_dict_suppliers:  # FIXME probably won't work with the current settings.
            def validation_suppliers(): return None
        else:
            if not isinstance(val_feed_dict_suppliers, dict) and len(self.val_error_dict.keys()) == 1:
                # cast validation supplier into a dict
                val_feed_dict_suppliers = {list(self.val_error_dict.keys())[0]: val_feed_dict_suppliers}

        # compute alpha_T using the validation set
        [tf.variables_initializer([self.p_dict[ve]]).run(feed_dict=data_supplier())
         for ve, data_supplier in val_feed_dict_suppliers.items()]

        # set hyper-derivatives to 0
        hyper_derivatives = self._initialize_hyper_derivatives_res()  # TODO deal better with the hyper-derivatives
        ss = tf.get_default_session()

        if summary_utils: summary_utils.run(ss, T)

        for _ in range(T - 1, -1, -1):

            # revert w_t to w_(t-1)
            ss.run(self._back_hist_op, feed_dict={self._w_placeholder: self.w_hist[_ - 1]})

            fds = train_feed_dict_supplier(self.global_step.eval())
            # TODO read below
            """ Unfortunately it looks like that the following two lines cannot be run together (will this
            degrade the performances???

            Furthermore, when using ADAM as parameter optimizer it looks like p_0 = nan (note that p_0 usually is not
            needed for the derivation of the hyper-gradients, unless w_0 itself
            depends from some hyper-parameter in hyper_list). Anyway should look into it"""

            if check_if_zero:
                if self._abs_sum_p.eval() < 1.e-20:
                    # ss.run([self.bk_ops, self.global_step.decrease], feed_dict=fds)
                    print('exiting backward pass at iteration %d.' % _)
                    return {k: list(reversed(v)) for k, v in hyper_derivatives.items()}

            # compute partial results for hyper_derivatives: alpha_t*B_t and concatenates them
            mid_res = ss.run([e[1] for e in self.hyper_derivatives], feed_dict=fds)
            for k in range(len(mid_res)):
                hyper_list = self.hyper_derivatives[k][0]
                mr = mid_res[k]
                for j in range(len(hyper_list)):
                    hyper_derivatives[hyper_list[j]].append(mr[j])

            # computes alpha_t = alpha_(t+1)*A_(t+1)
            ss.run([self._bk_ops, self.global_step.decrease], feed_dict=fds)

            if summary_utils: summary_utils.run(ss, _)

        hyper_derivatives = {k: list(reversed(v)) for k, v in hyper_derivatives.items()}

        # updates also variables that keep track of hyper-gradients
        [self._hyper_assign_ops[h].eval(feed_dict={self._grad_wrt_hypers_placeholder: ghv})
         for h, ghv in ReverseHyperGradient.std_collect_hyper_gradients(hyper_derivatives).items()]

        return hyper_derivatives

    def _initialize_hyper_derivatives_res(self):
        return {hyper: [] for hyper in self.hyper_list}

    def run_all(self, T, train_feed_dict_supplier=None, val_feed_dict_suppliers=None,
                forward_su=None, backward_su=None, after_forward_su=None, check_if_zero=False):
        """
        Performs both forward and backward step. See functions `forward` and `backward` for details.

        :param T:                   Total number of iterations
        :param train_feed_dict_supplier:   (feed_dict) supplier for training stage
        :param val_feed_dict_suppliers: (feed_dict) supplier for validation stage
        :param forward_su:          (optional) utils object with function `run` passed to `forward`
        :param backward_su:         (optional) utils object with function `run` passed to `backward`
        :param after_forward_su:    (optional) utils object with function `run` executed after `forward` and before
                                    `backward`
        :param check_if_zero:       (debug flag).
        :return: A dictionary of lists of step-wise hyper-gradients. In usual application the "true" hyper-gradients
                 can be obtained with method `std_collect_hyper_gradients`
        """
        self.initialize()
        self.forward(T, train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=forward_su)
        final_w = self.w_t.eval()

        if after_forward_su:
            after_forward_su.run(tf.get_default_session(), T)

        raw_hyper_grads = self.backward(
            T, val_feed_dict_suppliers=val_feed_dict_suppliers,
            train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=backward_su,
            check_if_zero=check_if_zero
        )

        tf.get_default_session().run(self._back_hist_op, feed_dict={self._w_placeholder: final_w})  # restore weights

        return raw_hyper_grads

    def run_all_truncated(self, T, train_feed_dict_supplier=None, val_feed_dict_suppliers=None,
                          forward_su=None, backward_su=None, after_forward_su=None, check_if_zero=False,
                          n_steps_truncated=None, opt_hyper_dicts=None):
        """
        Performs both forward and backward step. See functions `forward` and `backward` for details.

        :param n_steps_truncated: number of steps for the truncated backprop through time
        :param T:                   Total number of iterations
        :param train_feed_dict_supplier:   (feed_dict) supplier for training stage
        :param val_feed_dict_suppliers: (feed_dict) supplier for validation stage
        :param forward_su:          (optional) utils object with function `run` passed to `forward`
        :param backward_su:         (optional) utils object with function `run` passed to `backward`
        :param after_forward_su:    (optional) utils object with function `run` executed after `forward` and before
                                    `backward`
        :param check_if_zero:       (debug flag).
        :return: A dictionary of lists of step-wise hyper-gradients. In usual application the "true" hyper-gradients
                 can be obtained with method `std_collect_hyper_gradients`
        """
        assert n_steps_truncated is not None and opt_hyper_dicts is not None, 'wrong use of truncated backprop!,' \
                                                       ' all the arguments after n_steps_truncated must be given'

        k = n_steps_truncated if n_steps_truncated is not None else T
        n_updates = T // k

        self.initialize()

        for i in range(n_updates):
            self.forward(k, train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=forward_su)
            final_w = self.w_t.eval()
            if after_forward_su:
                after_forward_su.run(tf.get_default_session(), k*(i+1) - 1)

            last_global_step = self.global_step.eval()

            row_gradients = self.backward(
                k, val_feed_dict_suppliers=val_feed_dict_suppliers,
                train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=backward_su,
                check_if_zero=check_if_zero
            )

            # restore w and global_step to be the same after forward:
            tf.get_default_session().run(self._back_hist_op,
                                         feed_dict={self._w_placeholder: final_w})
            tf.get_default_session().run(self.global_step.assign_op,
                                         feed_dict={self.global_step.gs_placeholder: last_global_step})

            if opt_hyper_dicts is not None:
                hgs = ReverseHyperGradient.std_collect_hyper_gradients(row_gradients)
                ss = tf.get_default_session()
                for hyp in self.hyper_list:
                    ss.run(opt_hyper_dicts[hyp].assign_ops, feed_dict={self._grad_wrt_hypers_placeholder: hgs[hyp]})

        return row_gradients

    @staticmethod
    def std_collect_hyper_gradients(row_gradients):
        """
        Sums over all the step-wise hyper-gradients.

        :param row_gradients: Result of method `backward`
        :return: Hyper-gradients of validation error w.r.t. "fixed" hyperparameters
        """
        return {hyp: sum([r for r in res[1:]], res[0]) for hyp, res in row_gradients.items()}


class ForwardHyperGradient:

    def __init__(self, optimizer, hyper_dict, global_step=None):
        """
        Creates a new object that computes the hyper-gradient of validation errors in forward mode.
        See section 3.2 of Forward and Reverse Gradient-Based Hyperparameter Optimization
        (https://arxiv.org/abs/1703.01785)
        Note that this class only computes the hyper-gradient and does not perform hyperparameter optimization.

        :param optimizer: instance of Optimizer class, which represent the dynamics with which the model parameters are
                            updated
        :param hyper_dict: A dictionary of `{validation_error: hyper_pairs_list}` where
                            `validation_error` is a scalar tensor and `hyper_pairs_list` is single or a list of
                            pairs (hyperparameter, derivative_of_dynamics_w.r.t hyperparameter)
                            (matrix B_t in the paper). Unfortunately tensorflow does not computes Jacobians
                            efficiently yet (suggestions or pointer are welcomed)
        :param global_step: (optional) instance of `GlobalStep` to keep track of the optimization step
        """
        assert isinstance(optimizer, Optimizer)

        self.w = optimizer.raw_w  # might be variable or MergedVariable (never tested on Variables actually) ...
        self.w_t = MergedVariable.get_tensor(self.w)  # this is always a tensor

        self.tr_dynamics = optimizer.dynamics

        assert isinstance(hyper_dict, dict), '%s not allowed type. Should be a dict of (tf.Tensor,' \
                                             'list[(hyper-parameter, d_dynamics_d_hyper-parameter)]' % hyper_dict

        self.hyper_list = []  # more comfortable to use
        self.d_dynamics_d_hypers = []
        self.hyper_dict = {}  # standardizes hyper_dict parameter
        for k, v in hyper_dict.items():
            list_v = as_list(v)
            assert isinstance(list_v[0], tuple), "Something's wrong in hyper_dict %s, at least in entry%s. Check!"\
                                                 % (hyper_dict, list_v[0])
            self.hyper_dict[k] = list_v  # be sure values are lists!
            self.hyper_list += [pair[0] for pair in list_v]
            self.d_dynamics_d_hypers += [pair[1] for pair in list_v]

        self.val_errors = []  # will follow the same order as hyper_list
        for hyp in self.hyper_list:  # find the right validation error for hyp!
            for k, v in hyper_dict.items():
                all_hypers = [pair[0] for pair in as_list(v)]
                if hyp in all_hypers:
                    self.val_errors.append(k)
                    break

        for i, der in enumerate(self.d_dynamics_d_hypers):  # this automatic casting at the moment works only for SGD
            if not isinstance(der, ZMergedMatrix):
                print('Try casting d_dynamics_d_hyper to ZMergedMatrix')
                self.d_dynamics_d_hypers[i] = ZMergedMatrix(der)
                print('Successful')

        with self.w_t.graph.as_default():
            # global step
            self.global_step = global_step or GlobalStep()

            self.fw_ops = self.w.assign(self.tr_dynamics)  # TODO add here when hypers are sequence

            with tf.name_scope('direct_HO'):

                '''
                Creates one z per hyper-parameter and assumes that each hyper-parameter is a vector
                '''
                self.zs = [self._create_z(hyp) for hyp in self.hyper_list]

                self.zs_dynamics = [optimizer.jac_z(z) + dd_dh
                                    for z, dd_dh in zip(self.zs, self.d_dynamics_d_hypers)]

                print('z dynamics', self.zs_dynamics[0])
                print('z', self.zs[0])

                self.zs_assigns = [z.assign(z_dyn) for z, z_dyn in zip(self.zs, self.zs_dynamics)]

                self.grad_val_err = [tf.gradients(v_e, self.w_t)[0] for v_e in self.val_errors]
                assert all([g is not None for g in self.grad_val_err]), 'Some gradient of the validation error is None!'

                self.grad_wrt_hypers = [dot(gve, z.tensor) for z, gve in zip(self.zs, self.grad_val_err)]

                with tf.name_scope('hyper_gradients'):  # ADDED 28/3/17 keeps track of hyper-gradients as tf.Variable
                    self.hyper_gradient_vars = [tf.Variable(tf.zeros_like(hyp), name=simple_name(hyp))
                                                for hyp in self.hyper_list]
                    self.hyper_gradients_dict = {hyp: hgv for hyp, hgv  # redundant.. just for comfort ..
                                                 in zip(self.hyper_list, self.hyper_gradient_vars)}
                    self._hyper_assign_ops = [v.assign(ght)
                                              for v, ght in zip(self.hyper_gradient_vars, self.grad_wrt_hypers)]

    def get_reverse_hyper_dict(self):
        """

        :return: A dictionary of (validation errors, (list of) hyper-parameters)
                 suitable as input for `ReverseHyperGradient` initializer.
        """
        return {k: [e[0] for e in v] for k, v in self.hyper_dict.items()}

    def _create_z(self, hyper):
        """
        Initializer for Z-variables. Used internally.

        :param hyper:
        :return:
        """
        shape_h = hyper.get_shape().as_list()
        assert len(shape_h) < 2, 'only scalar or vector hyper-parameters are accepted: %s shape: %s' % (hyper, shape_h)
        dim_h = shape_h[0] if shape_h else 1

        components = self.w.var_list(Vl_Mode.TENSOR) if isinstance(self.w, MergedVariable) else [self.w_t]

        print('components', components)

        with tf.name_scope('z'):

            z_components = [tf.Variable(tf.zeros([c.get_shape().as_list()[0], dim_h]), name=hyper.name.split(':')[0])
                            for c in components]
            mvz = ZMergedMatrix(z_components)
            print(mvz.tensor)
            return mvz

    def initialize(self):
        """
        Helper for initializing all the variables. Builds and runs model variables, Zs and global step initializers.

        :return: None
        """
        assert tf.get_default_session() is not None, 'No default tensorflow session!'
        var_init = self.w.var_list(Vl_Mode.BASE) if isinstance(self.w, MergedVariable) else [self.w]
        tf.variables_initializer(var_init + self.hyper_gradient_vars + [self.global_step.var]).run()
        [tf.variables_initializer(z.components).run() for z in self.zs]

    def step_forward(self, train_feed_dict_supplier=None, summary_utils=None):
        """
        Updates for one step both model parameters (according to the optimizer dynamics) and
        Z-variables.

        :param train_feed_dict_supplier: (optional) A callable with signature `t ->  feed_dict` to pass to
                                        `tf.Session.run`
                                    feed_dict argument
        :param summary_utils: (optional) object with method `run(session, iteration)`
        :return: None
        """

        if not train_feed_dict_supplier:
            # noinspection PyUnusedLocal
            def feed_dict_supplier(step=None): return None

        ss = tf.get_default_session()

        fd = train_feed_dict_supplier(self.global_step.eval())

        ss.run(self.zs_assigns, feed_dict=fd)
        ss.run(self.fw_ops, feed_dict=fd)
        if summary_utils:
            summary_utils.run(ss, self.global_step.eval())
        self.global_step.increase.eval()

    def hyper_gradients(self, val_feed_dict_supplier=None, new_mode=False):
        """
        Method that computes the hyper-gradient.

        :param new_mode:
        :param val_feed_dict_supplier: single  supplier or list of suppliers for the examples in the validation set

        :return: Dictionary: {hyper-parameter: hyper-gradient} or None in new_mode
        """
        if not isinstance(val_feed_dict_supplier, dict) and len(self.hyper_list) == 1:  # fine if there is only 1 hyper
            val_feed_dict_supplier = {self.hyper_list[0]: val_feed_dict_supplier}

        val_sup_lst = []
        for hyp in self.hyper_list:  # find the right validation error for hyp!
            for k, v in self.hyper_dict.items():
                all_hypers = [e[0] for e in v]
                if hyp in all_hypers:
                    val_sup_lst.append(val_feed_dict_supplier[k])
                    break

        [assign_op.eval(feed_dict=vsl(self.global_step.eval())) for
         assign_op, vsl in zip(self._hyper_assign_ops, val_sup_lst)]

        if not new_mode:
            print("WARNING: things with hyper-gradients have been changed. Now they are treated as variables that "
                  "is much easier.")
            computations = [gve.eval(feed_dict=vsl(self.global_step.eval())) for
                            gve, vsl in zip(self.grad_wrt_hypers, val_sup_lst)]  # computes the actual gradients

            def cast_to_scalar_if_needed(grad):
                sh_grad = list(np.shape(grad))
                return grad[0] if len(sh_grad) == 1 and sh_grad[0] == 1 else grad

            return {hyp: cast_to_scalar_if_needed(gdd) for hyp, gdd in zip(self.hyper_list, computations)}

    def run_all(self, T, train_feed_dict_supplier=None, val_feed_dict_suppliers=None,
                forward_su=None, after_forward_su=None):

        self.initialize()
        for k in range(T):
            self.step_forward(train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=forward_su)

        if after_forward_su:
            after_forward_su.run(tf.get_default_session(), T)

        return self.hyper_gradients(val_feed_dict_supplier=val_feed_dict_suppliers)


class RealTimeHO:

    def __init__(self, forward_hyper_grad, hyperparameter_optimizers, hyper_projections=None, hyper_step=None):
        """
        Helper class to perform Real Time Hyperparameter optimization.
        See section 3.3 of Forward and Reverse Gradient-Based Hyperparameter Optimization
        (https://arxiv.org/abs/1703.01785)

        :param forward_hyper_grad:          instance of `ForwardHyperGradient`. Used to compute hyper-gradients
        :param hyperparameter_optimizers:   single or list of Optimizer for the hyper-parameter descent procedure
        :param hyper_projections:           (optional) list of assign ops that performs projection to
                                            onto a convex subset of the hyperparameter space.
        :param hyper_step:                  (optional) instance of `GlobalStep` class that keeps tracks of the number
                                            of hyper-batches performed so far.
        """
        assert isinstance(forward_hyper_grad, ForwardHyperGradient)
        self.direct_doh = forward_hyper_grad

        assert isinstance(hyperparameter_optimizers, (list, Optimizer)), "hyper_opt_dicts should be a single " \
                                                                         "Optimizer or a list of Optimizer. Instead" \
                                                                         "is %s" % hyperparameter_optimizers
        self.hyper_opt_dicts = as_list(hyperparameter_optimizers)

        self.hyper_projections = hyper_projections or []

        self.hyper_step = hyper_step or GlobalStep()

        # self.collected_hyper_gradients = {}

    def initialize(self):
        tf.variables_initializer(self.direct_doh.hyper_list + [self.hyper_step.var]).run()
        [ahd.support_variables_initializer().run() for ahd in self.hyper_opt_dicts]
        self.direct_doh.initialize()

    def hyper_batch(self, hyper_batch_size, train_feed_dict_supplier=None, val_feed_dict_suppliers=None,
                    saver=None, collect_data=True, apply_hyper_gradients=True):
        """
        Executes an entire hyper-batch.

        :param hyper_batch_size: size of the hyper-batch (number of elementary iterations)
        :param train_feed_dict_supplier: supplier for the training stage
        :param val_feed_dict_suppliers: dictionary of suppliers for the validation stage
        :param saver: (optional) callable to save some statistics
        :param collect_data: (optional) flag for saving training data
        :param apply_hyper_gradients: (debug flag, default `True`) if `False` not hyperparameter update is performed
        :return:
        """
        ss = tf.get_default_session()

        for k in range(hyper_batch_size):
            self.direct_doh.step_forward(train_feed_dict_supplier=train_feed_dict_supplier)

        # compute hyper_grads
        self.direct_doh.hyper_gradients(val_feed_dict_supplier=val_feed_dict_suppliers, new_mode=True)

        if apply_hyper_gradients:
            [ss.run(hod.assign_ops) for hod in self.hyper_opt_dicts]
            [ss.run(prj) for prj in self.hyper_projections]

        self.hyper_step.increase.eval()

        if saver:
            saver(ss, self.hyper_step.eval(), collect_data=collect_data)  # saver should become a class...


def create_hyperparameter_optimizers(rf_hyper_gradients, optimizer_class, **optimizers_kw_args):  # TODO review this m
    """
    Helper for creating descent procedure for hyperparameters

    :param rf_hyper_gradients: instance of `ForwardHyperGradient` or `ReverseHyperGradient` class
    :param optimizer_class:  callable for instantiating the single optimizers
    :param optimizers_kw_args: arguments to pass to `optimizer_creator`
    :return: List of `Optimizer` objects
    """
    # assert isinstance(optimizer_class, Optimizer), '%s should be an Optimizer' % optimizer_class
    return [optimizer_class.create(hyp, **optimizers_kw_args, grad=hg, w_is_state=False)
            for hyp, hg in zip(rf_hyper_gradients.hyper_list, rf_hyper_gradients.hyper_gradient_vars)]


def positivity(hyper_list):
    return [hyp.assign(tf.maximum(hyp, tf.zeros_like(hyp))) for hyp in hyper_list]


def print_hyper_gradients(hyper_gradient_dict):
    for k, v in hyper_gradient_dict.items():
        print(k.name, v)
