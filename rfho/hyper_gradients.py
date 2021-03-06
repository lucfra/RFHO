"""
This module contains the core classes of the package that implement the three hyperparameter optimization methods
presented in Forward and Reverse Gradient-Based Hyperparameter Optimization (https://arxiv.org/abs/1703.01785).
"""

# TODO put tf.Session optional parameter in all the methods that require `run`

# import numpy as np
import tensorflow as tf

from rfho.optimizers import Optimizer, AdamOptimizer
from rfho.utils import dot, MergedVariable, VlMode, as_list, simple_name, GlobalStep, ZMergedMatrix, flatten_list
from rfho.utils import call_method_optional_param as cmo


class ReverseHG:
    """
    Class to compute hyper-gradients in reverse mode
    """

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
        self.w_t = self.w # MergedVariable.get_tensor(self.w)  # this is always a tensor

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

            self._fw_ops = optimizer.assign_ops  # add here when hyper-parameters are sequence

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
                        any([isinstance(v, MergedVariable) for v in self.w.var_list(VlMode.RAW)]):
                    state_components = self.w.var_list(VlMode.TENSOR)

                    # equation (8)
                    self.p_dynamics = {ve: tf.concat(tf.gradients(lagrangian, state_components), 0)
                                       for ve, lagrangian in self.lagrangians_dict.items()}
                else:
                    # equation (8)
                    self.p_dynamics = {ve: tf.gradients(lagrangian, self.w_t)[0]
                                       for ve, lagrangian in self.lagrangians_dict.items()}  # equation (7)

                self._bk_ops = [self.p_dict[ve].assign(self.p_dynamics[ve])
                                for ve in self.val_error_dict]  # add here when hp are sequ.

            with tf.name_scope('w_history_ops'):
                self._w_placeholder = tf.placeholder(self.w_t.dtype)

                self._back_hist_op = self.w.assign(self._w_placeholder)

            with tf.name_scope('hyper_derivatives'):
                # equation (10) without summation.
                self.hyper_derivatives = [
                    (self.val_error_dict[ve], tf.gradients(lagrangian, self.val_error_dict[ve]))
                    for ve, lagrangian in self.lagrangians_dict.items()
                ]  # list of couples (hyper_list, list of tensors hyper_gradients)  (lists are unhashable!)
                # check that all hyper-gradients are defined
                assert all(e is not None for e in flatten_list(
                    [e[1] for e in self.hyper_derivatives])), 'Some gradient of the validation error is None!'

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

    def initialize(self, session=None):
        """
        Helper for initializing all the variables. Builds and runs model variables and global step initializers.
        Note that dual variables are initialized only when calling `backward`.
        
        :param session: optional tensorflow session (if None default session is used) 
        
        :return: None
        """
        ss = session or tf.get_default_session()
        assert ss, 'No default tensorflow session!'
        if isinstance(self.w, MergedVariable):
            self.w.initialize(session=session)
        else:
            ss.run(tf.variables_initializer([self.w]))
        ss.run(tf.variables_initializer(self.hyper_gradient_vars + [self.global_step.var]))

    def forward(self, T, train_feed_dict_supplier=None, summary_utils=None):
        """
        Performs (forward) optimization of the parameters.

        :param T: Total number of iterations
        :param train_feed_dict_supplier: (optional) A callable with signature `t -> feed_dict`
                                            or `() -> feed_dict` to pass to
                                            `tf.Session.run` feed_dict argument
        :param summary_utils: (optional) object that implements a method method `run(tf.Session, step)`
                                that is executed at every iteration (see for instance utils.PrintUtils
        :return: None
        """
        if not train_feed_dict_supplier:
            train_feed_dict_supplier = lambda: None

        # var_init = self.w.var_list(VlMode.BASE) if isinstance(self.w, MergedVariable) else [self.w]
        # tf.variables_initializer(var_init + [self.global_step.var]).run()

        ss = tf.get_default_session()
        self.w_hist.clear()

        for t in range(T):
            self.w_hist.append(self.w_t.eval())
            ss.run([self._fw_ops], feed_dict=cmo(train_feed_dict_supplier, self.global_step.eval()))
            self.global_step.increase.eval()
            if summary_utils:
                summary_utils.run(ss, t)

    def backward(self, T, val_feed_dict_suppliers=None, train_feed_dict_supplier=None, hyper_batch_step=None,
                 summary_utils=None, check_if_zero=False):
        """
        Performs backward computation of hyper-gradients

        :param hyper_batch_step: supports for stochastic sampling of validation set
        :param T: Total number of iterations
        :param val_feed_dict_suppliers: either a callable that returns a feed_dict
                                        or a dictionary {validation_error tensor: callable
                                            (signature `t -> feed_dict` or `() -> feed_dict`) that
                                        is used to initialize the dual variables `p` (generally supplier of
                                        validation example set).

        :param train_feed_dict_supplier: (optional) A callable with signature `t -> feed_dict` or `() -> feed_dict` to
                                            pass to `tf.Session.run`
                                    feed_dict argument
        :param summary_utils: (optional) object that implements a method method `run(tf.Session, step)`
                                that is executed at every iteration (see for instance utils.PrintUtils
        :param check_if_zero: (optional) debug flag
        :return: A dictionary of lists of step-wise hyper-gradients. In usual application the "true" hyper-gradients
                 can be obtained with method std_collect_hyper_gradients
        """
        if not train_feed_dict_supplier:
            # noinspection PyUnusedLocal
            train_feed_dict_supplier = lambda: None
        if not val_feed_dict_suppliers:  # FIXME probably won't work with the current settings.
            val_feed_dict_suppliers = lambda: None
        else:
            if not isinstance(val_feed_dict_suppliers, dict) and len(self.val_error_dict.keys()) == 1:
                # cast validation supplier into a dict
                val_feed_dict_suppliers = {list(self.val_error_dict.keys())[0]: val_feed_dict_suppliers}

        # compute alpha_T using the validation set
        [tf.variables_initializer([self.p_dict[ve]]).run(feed_dict=cmo(data_supplier, hyper_batch_step))
         for ve, data_supplier in val_feed_dict_suppliers.items()]

        # set hyper-derivatives to 0
        hyper_derivatives = self._initialize_hyper_derivatives_res()  # TODO deal better with the hyper-derivatives
        ss = tf.get_default_session()

        if summary_utils: summary_utils.run(ss, T)

        for t in range(T - 1, -1, -1):
            self.global_step.decrease.eval()

            # revert w_t to w_(t-1)
            ss.run(self._back_hist_op, feed_dict={self._w_placeholder: self.w_hist[t]})

            # noinspection PyNoneFunctionAssignment
            fds = cmo(train_feed_dict_supplier, self.global_step.eval())
            # TODO read below  (maybe use tf.control_dependencies ... would it speed this up?)
            """ Unfortunately it looks like that the following two lines cannot be run together (will this
            degrade the performances???"""

            if check_if_zero:  # debug
                if self._abs_sum_p.eval() < 1.e-20:
                    # ss.run([self.bk_ops, self.global_step.decrease], feed_dict=fds)
                    print('exiting backward pass at iteration %d.' % t)
                    return {k: list(reversed(v)) for k, v in hyper_derivatives.items()}

            # compute partial results for hyper_derivatives: alpha_t*B_t and concatenates them
            mid_res = ss.run([e[1] for e in self.hyper_derivatives], feed_dict=fds)
            for k in range(len(mid_res)):
                hyper_list = self.hyper_derivatives[k][0]
                mr = mid_res[k]
                for j in range(len(hyper_list)):
                    hyper_derivatives[hyper_list[j]].append(mr[j])

            # computes alpha_t = alpha_(t+1)*A_(t+1)

            ss.run([self._bk_ops], feed_dict=fds)  # check this global_step here.. (for Adam)

            if summary_utils: summary_utils.run(ss, t)

        hyper_derivatives = {k: list(reversed(v)) for k, v in hyper_derivatives.items()}

        # updates also variables that keep track of hyper-gradients
        [self._hyper_assign_ops[h].eval(feed_dict={self._grad_wrt_hypers_placeholder: ghv})
         for h, ghv in ReverseHG.std_collect_hyper_gradients(hyper_derivatives).items()]

        return hyper_derivatives

    def _initialize_hyper_derivatives_res(self):
        return {hyper: [] for hyper in self.hyper_list}

    def run_all(self, T, train_feed_dict_supplier=None, val_feed_dict_suppliers=None, hyper_batch_step=None,
                forward_su=None, backward_su=None, after_forward_su=None, check_if_zero=False):
        """
        Performs both forward and backward step. See functions `forward` and `backward` for details.

        :param hyper_batch_step: support for stochastic sampling of validation set
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
        # self.initialize()
        self.forward(T, train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=forward_su)
        final_w = self.w_t.eval()
        last_global_step = self.global_step.eval()

        if after_forward_su:
            after_forward_su.run(tf.get_default_session(), T)

        raw_hyper_grads = self.backward(
            T, val_feed_dict_suppliers=val_feed_dict_suppliers,
            train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=backward_su,
            check_if_zero=check_if_zero,
            hyper_batch_step=hyper_batch_step
        )

        tf.get_default_session().run(self._back_hist_op,
                                     feed_dict={self._w_placeholder: final_w})  # restore weights

        tf.get_default_session().run(self.global_step.assign_op,
                                     feed_dict={self.global_step.gs_placeholder: last_global_step})

        return raw_hyper_grads

    # TODO To Riccardo: if useless please delete
    def run_all_truncated(self, T, train_feed_dict_supplier=None, val_feed_dict_suppliers=None,
                          forward_su=None, backward_su=None, after_forward_su=None, check_if_zero=False,
                          n_steps_truncated=None, opt_hyper_dicts=None):
        """
        Performs both forward and backward step. See functions `forward` and `backward` for details.

        :param opt_hyper_dicts:
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
        assert n_steps_truncated is not None and opt_hyper_dicts is not None, 'wrong use of truncated reverse-HO!,' \
                                                                              ' all the arguments ' \
                                                                              'after n_steps_truncated must be given'

        k = n_steps_truncated if n_steps_truncated is not None else T
        n_updates = T // k

        self.initialize()

        raw_gradients = None
        for i in range(n_updates):
            raw_gradients = self.run_all(k, train_feed_dict_supplier, val_feed_dict_suppliers, forward_su, backward_su,
                                         after_forward_su, check_if_zero)

            if opt_hyper_dicts is not None:
                hgs = ReverseHG.std_collect_hyper_gradients(raw_gradients)
                ss = tf.get_default_session()
                for hyp in self.hyper_list:
                    ss.run(opt_hyper_dicts[hyp].assign_ops, feed_dict={self._grad_wrt_hypers_placeholder: hgs[hyp]})

        return raw_gradients

    @staticmethod
    def std_collect_hyper_gradients(row_gradients):
        """
        Sums over all the step-wise hyper-gradients.

        :param row_gradients: Result of method `backward`
        :return: Hyper-gradients of validation error w.r.t. "fixed" hyperparameters
        """
        return {hyp: sum([r for r in res[1:]], res[0]) for hyp, res in row_gradients.items()}


class ForwardHG:
    """
    Computes the hyper-gradient in forward mode
    """

    def __init__(self, optimizer, hyper_dict, global_step=None, devices=None):
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
        self.w_t = self.w # MergedVariable.get_tensor(self.w)  # this is always a tensor

        self.tr_dynamics = optimizer.dynamics

        assert isinstance(hyper_dict, dict), '%s not allowed type. Should be a dict of (tf.Tensor,' \
                                             'list[(hyper-parameter, d_dynamics_d_hyper-parameter)]' % hyper_dict

        self.hyper_list = []  # more comfortable to use
        self.d_dynamics_d_hypers = []
        self.hyper_dict = {}  # standardizes hyper_dict parameter
        self._inverse_hyper_dict = {}  # hyperparameter-validation error pairs
        for k, v in hyper_dict.items():
            list_v = as_list(v)
            # assert isinstance(list_v[0], tuple), "Something's wrong in hyper_dict %s, at least in entry%s. Check!"\
            #                                      % (hyper_dict, list_v[0])
            self.hyper_dict[k] = list_v  # be sure values are lists!
            self._inverse_hyper_dict = {**self._inverse_hyper_dict, **{hyp: k for hyp in list_v}}
            self.hyper_list += [pair[0] if isinstance(pair, (tuple, list)) else pair for pair in list_v]
            self.d_dynamics_d_hypers += [pair[1] if isinstance(pair, (tuple, list)) else
                                         optimizer.auto_d_dynamics_d_hyper(pair)  # try to compute it automatically
                                         for pair in list_v]

        self.val_errors = []  # will follow the same order as hyper_list
        for hyp in self.hyper_list:  # find the right validation error for hyp!
            for k, v in hyper_dict.items():
                all_hypers = [pair[0] if isinstance(pair, (list, tuple)) else pair for pair in as_list(v)]
                if hyp in all_hypers:
                    self.val_errors.append(k)
                    break

        for i, der in enumerate(self.d_dynamics_d_hypers):  # this automatic casting at the moment works only for SGD
            if not isinstance(der, ZMergedMatrix):
                print('Try casting d_dynamics_d_hyper to ZMergedMatrix')
                self.d_dynamics_d_hypers[i] = ZMergedMatrix(der)
                print('Successful')

        devices = as_list(devices)  # at most will be [None]

        with self.w_t.graph.as_default():
            # global step
            self.global_step = global_step or GlobalStep()

            self.fw_ops = optimizer.assign_ops  # add here when hypers are sequence (...)

            with tf.name_scope('ForwardHG'):
                '''
                Creates one z per hyper-parameter and assumes that each hyper-parameter is a vector
                '''
                self.grad_wrt_hypers, self.zs, self.zs_dynamics, self._zs_assigns = [], [], [], []
                self.hyper_gradient_vars, self._hyper_assign_ops = [], []

                self.grad_val_err = {ve: tf.identity(tf.gradients(ve, self.w_t)[0],
                                                      name='grad_val_err_%s' % simple_name(ve.name))
                                     for ve in self.hyper_dict.keys()}
                self._gve_inv_dict = {hyp: self.grad_val_err[ve] for hyp, ve in self._inverse_hyper_dict.items()}

                for k, hyp in enumerate(self.hyper_list):
                    with tf.device(devices[k % len(devices)]):
                        self.zs.append(self._create_z(hyp))

                        with tf.name_scope('Z_dynamics'):
                            self.zs_dynamics.append(optimizer.jac_z(self.zs[k]) + self.d_dynamics_d_hypers[k])
                            self._zs_assigns.append(self.zs[k].assign(self.zs_dynamics[k]))

                        self.grad_wrt_hypers.append(dot(self._gve_inv_dict[hyp], self.zs[k], name='hyper_grad_wrt_h'))

                        with tf.name_scope('hyper_gradients'):
                            self.hyper_gradient_vars.append(tf.Variable(tf.zeros_like(hyp), name=simple_name(hyp)))
                            self._hyper_assign_ops.append(self.hyper_gradient_vars[k].assign(self.grad_wrt_hypers[k]))

                # final operations
                self.hyper_gradients_dict = {hyp: hgv for hyp, hgv  # redundant.. just for comfort ..
                                             in zip(self.hyper_list, self.hyper_gradient_vars)}
                # hyper-gradient check
                assert all([g is not None for g in self.grad_val_err]), 'Some gradient ' \
                                                                        'of the validation error is None!'

    def get_reverse_hyper_dict(self):
        """

        :return: A dictionary of (validation errors, (list of) hyper-parameters)
                 suitable as input for `ReverseHG` initializer.
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

        components = self.w.var_list(VlMode.TENSOR) if isinstance(self.w, MergedVariable) else [self.w_t]

        # print('components', components)

        z_components = [tf.Variable(tf.zeros([c.get_shape().as_list()[0], dim_h]),
                                    name=simple_name(hyper))
                        for c in components]
        mvz = ZMergedMatrix(z_components, name='Z_' + simple_name(hyper))
        return mvz

    def initialize(self, session=None):
        """
        Helper for initializing all the variables. Builds and runs model variables, 
        Zs and global step initializers.
        
        :param session: optional tensorflow session (if None default session is used) 
        
        :return: None
        """
        ss = session or tf.get_default_session()
        assert ss, 'No default tensorflow session!'
        if isinstance(self.w, MergedVariable):
            self.w.initialize(session=session)
        else:
            ss.run(tf.variables_initializer([self.w]))  # never tested
        ss.run(tf.variables_initializer(self.hyper_gradient_vars + [self.global_step.var]))
        [z.initializer().run() for z in self.zs]
        return True

    def step_forward(self, train_feed_dict_supplier=None, summary_utils=None):
        """
        Updates for one step both model parameters (according to the optimizer dynamics) and
        Z-variables.

        :param train_feed_dict_supplier: (optional) A callable with signature `t ->  feed_dict` to pass to
                                        `tf.Session.run`
                                    feed_dict argument
        :param summary_utils: (optional) object with method `run(session, iteration)`
        :return: feed dictionary for this step
        """

        if not train_feed_dict_supplier:
            train_feed_dict_supplier = lambda: None

        ss = tf.get_default_session()

        # noinspection PyNoneFunctionAssignment
        fd = cmo(train_feed_dict_supplier, self.global_step.eval())

        ss.run(self._zs_assigns, feed_dict=fd)
        ss.run(self.fw_ops, feed_dict=fd)

        if summary_utils:  # TODO delete!
            summary_utils.run(ss, self.global_step.eval())

        self.global_step.increase.eval()
        return fd

    def hyper_gradients(self, val_feed_dict_supplier=None, hyper_batch_step=None):
        """
        Method that computes the hyper-gradient.

        :param hyper_batch_step: support for stochastic sampling of validation points
        :param val_feed_dict_supplier: single  supplier or list of suppliers for the examples in the validation set

        :return: Dictionary: {hyper-parameter: hyper-gradient} or None in new_mode
        """
        if not isinstance(val_feed_dict_supplier, dict) and len(self.hyper_list) == 1:  # fine if there is only 1 hyper
            val_feed_dict_supplier = {self.hyper_list[0]: val_feed_dict_supplier}

        val_sup_lst = []
        if val_feed_dict_supplier is None:
            val_sup_lst = [lambda: None] * len(self.hyper_list)
        else:
            for hyp in self.hyper_list:  # find the right validation error for hyp!
                for k, v in self.hyper_dict.items():
                    all_hypers = [e[0] if isinstance(e, (list, tuple)) else e for e in v]
                    if hyp in all_hypers:
                        val_sup_lst.append(val_feed_dict_supplier[k])
                        break

        # NEW VARIABLE-BASED HYPER-GRADIENTS
        [assign_op.eval(feed_dict=cmo(vsl, hyper_batch_step)) for
         assign_op, vsl in zip(self._hyper_assign_ops, val_sup_lst)]

        return {hyp: self.hyper_gradients_dict[hyp].eval() for hyp in self.hyper_list}

    def run_all(self, T, train_feed_dict_supplier=None, val_feed_dict_suppliers=None, hyper_batch_step=None,
                forward_su=None, after_forward_su=None):
        """
        Helper method for running

        :param hyper_batch_step: support for stochastic sampling of validation  set
        :param T:
        :param train_feed_dict_supplier:
        :param val_feed_dict_suppliers:
        :param forward_su:
        :param after_forward_su:
        :return:
        """

        # self.initialize()
        for k in range(T):
            self.step_forward(train_feed_dict_supplier=train_feed_dict_supplier, summary_utils=forward_su)

        if after_forward_su:
            after_forward_su.run(tf.get_default_session(), T)

        return self.hyper_gradients(val_feed_dict_suppliers, hyper_batch_step)


class HyperOptimizer:
    """
    Interface class for gradient-based hyperparameter optimization methods.
    """

    def __init__(self, optimizer, hyper_dict, method, hyper_grad_kwargs=None,
                 hyper_optimizer_class=AdamOptimizer, **optimizers_kwargs):
        """
        Interface instance of gradient-based hyperparameter optimization methods.

        :param optimizer: parameter optimization dynamics (obtained from `Optimizer.create` methods)
        :param hyper_dict: dictionary of validation errors and list of hyperparameters to be optimized
        :param method:  method with which to compute hyper-gradients: Forward
                        or Reverse-Ho
        :param hyper_grad_kwargs: dictionary of keyword arguments for `HyperGradient` classes (usually None)
        :param hyper_optimizer_class: (default Adam) Optimizer class for optimization of the hyperparameters
        :param optimizers_kwargs: keyword arguments for hyperparameter optimizers (like hyper-learning rate)
        """
        assert method in [ReverseHG, ForwardHG]
        assert hyper_optimizer_class is None or issubclass(hyper_optimizer_class, Optimizer)
        assert isinstance(hyper_dict, dict)
        assert isinstance(optimizer, Optimizer)

        if not hyper_grad_kwargs: hyper_grad_kwargs = {}
        self.hyper_iteration_step = GlobalStep(name='hyper_iteration_step')
        self._report_hyper_it_init = tf.report_uninitialized_variables([self.hyper_iteration_step.var])
        # self.hyper_batch_step = GlobalStep(name='hyper_batch_step')
        self.hyper_batch_step = GlobalStep(name='batch_step')

        # automatically links eventual optimizer global step (like in Adam) to HyperGradient global step
        hyper_grad_kwargs['global_step'] = hyper_grad_kwargs.get(
            'global_step', optimizer.global_step if hasattr(optimizer, 'global_step') else GlobalStep())

        # automatically links eventual hyper-optimizer global step (like in Adam) to batch_step
        if hyper_optimizer_class == AdamOptimizer:
            optimizers_kwargs['global_step'] = self.hyper_batch_step
            optimizers_kwargs.setdefault('eps', 1.e-14)

        self.hyper_gradients = method(optimizer, hyper_dict, **hyper_grad_kwargs)

        if hyper_optimizer_class:
            # noinspection PyTypeChecker
            self.hyper_optimizers = create_hyperparameter_optimizers(
                self.hyper_gradients, optimizer_class=hyper_optimizer_class, **optimizers_kwargs)
        else:
            self.hyper_optimizers = None

    @property
    def hyper_list(self):
        """

        :return: list of hyperparameters that are/will be optimized
        """
        return self.hyper_gradients.hyper_list

    def initialize(self, session=None, complete_reinitialize=False):
        """
        Initialize all tensorflow variables. This method has two behaviours:

        - first time it is called (after entering a Session run block) or when flag `complete_reinitialize` is `True`
            initializes all the relevant variables
        - subsequent times, reinitialize only model variables (next hyper-iteration).
        
        :param: complete_reinitialize: (default `False`) if True reinitialize hyper-step counts and hyperparameter
                                        optimizers regardless of
        :param: session: optional tensorflow session (if None default session is used) 

        :return: True if this is the first initialization
        """
        ss = tf.get_default_session()
        assert ss, 'No default session.'

        never_initialized = bool(self._report_hyper_it_init.eval())

        if complete_reinitialize or never_initialized:  # never initialized or subsequent run of
            # Session run block (for instance in a Ipython book)
            tf.variables_initializer(self.hyper_gradients.hyper_list).run()
            if self.hyper_optimizers:
                [opt.support_variables_initializer().run() for opt in self.hyper_optimizers]
            tf.variables_initializer([self.hyper_iteration_step.var, self.hyper_batch_step.var]).run()
        else:
            self.hyper_iteration_step.increase.eval()

        self.hyper_gradients.initialize(session=session)

        return never_initialized

    def run(self, T, train_feed_dict_supplier=None, val_feed_dict_suppliers=None,
            hyper_constraints_ops=None,
            _debug_no_hyper_update=False):  # TODO add session parameter
        """

        :param _debug_no_hyper_update: 
        :param T: number of steps
        :param train_feed_dict_supplier:
        :param val_feed_dict_suppliers:
        :param hyper_constraints_ops: (list of) either callable (no parameters) or tensorflow ops
        :return:
        """
        # idea: if steps == T then do full reverse, or forward, otherwise do trho and rtho
        # after all the main difference is that if we go with the full version, after the gradient has been
        # computed, the method `initialize()` is called.

        self.hyper_gradients.run_all(T, train_feed_dict_supplier=train_feed_dict_supplier,
                                     val_feed_dict_suppliers=val_feed_dict_suppliers,
                                     hyper_batch_step=self.hyper_batch_step.eval())
        if not _debug_no_hyper_update:
            [tf.get_default_session().run(hod.assign_ops) for hod in self.hyper_optimizers]
            if hyper_constraints_ops: [op() if callable(op) else op.eval()
                                       for op in as_list(hyper_constraints_ops)]

            self.hyper_batch_step.increase.eval()


def create_hyperparameter_optimizers(rf_hyper_gradients, optimizer_class, **optimizers_kw_args):
    """
    Helper for creating descent procedure for hyperparameters

    :param rf_hyper_gradients: instance of `ForwardHG` or `ReverseHG` class
    :param optimizer_class:  callable for instantiating the single optimizers
    :param optimizers_kw_args: arguments to pass to `optimizer_creator`
    :return: List of `Optimizer` objects
    """
    assert issubclass(optimizer_class, Optimizer), '%s should be an Optimizer' % optimizer_class
    return [optimizer_class.create(hyp, **optimizers_kw_args, grad=hg, w_is_state=False)
            for hyp, hg in zip(rf_hyper_gradients.hyper_list, rf_hyper_gradients.hyper_gradient_vars)]


def positivity(hyper_list):
    """
    Simple positivity constraints for a list of hyperparameters

    :param hyper_list: single variable or list of variable (hyperparameters)
    :return: single or list of assign ops, one for each variable in `hyper_list`
    """
    lst = [hyp.assign(tf.maximum(hyp, tf.zeros_like(hyp))) for hyp in as_list(hyper_list)]
    return lst if len(lst) > 1 else lst[0]


def print_hyper_gradients(hyper_gradient_dict):  # TODO to be removed
    """
    Old helper function to nicely print hyper-gradients

    :param hyper_gradient_dict:
    :return:
    """
    for k, v in hyper_gradient_dict.items():
        print(k.name, v)
