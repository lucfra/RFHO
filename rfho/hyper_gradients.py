import tensorflow as tf
import numpy as np

from rfho.optimizers import Optimizer
from rfho.utils import dot, MergedVariable, Vl_Mode, as_list, simple_name


class GlobalStep:
    """
    Helper for global step (probably would be present also in tensorflow)
    """

    def __init__(self, start_from=0):
        self._var = tf.Variable(start_from, trainable=False, name='global_step')
        self.increase = self.var.assign_add(1)
        self.decrease = self.var.assign_sub(1)

    def eval(self, auto_initialize=True):
        if not auto_initialize:
            return self.var.eval()
        else:
            try:
                return self.var.eval()
            except tf.errors.FailedPreconditionError:
                tf.variables_initializer([self.var]).run()
                return self.var.eval()

    @property
    def var(self):
        return self._var


class ReverseHyperGradient:

    # noinspection SpellCheckingInspection
    def __init__(self, optimizer, hyper_dict, state_history=None, global_step=None):
        """
        Creates a new object that computes the hyper-gradient of validation errors in reverse mode.
        See section 3.1 of Forward and Reverse Gradient-Based Hyperparameter Optimization
        (https://arxiv.org/abs/1703.01785)
        Note that this class only computes the hyper-gradient and does not perform hyperparameter optimization.

        :param optimizer: insance of Optimizer class, which represent the dynamics with which the model parameters are
                            updated
        :param hyper_dict: A dictionary of `{validation_error: hyperparameter or list_of_hyperparameters}` where
                            `validation_error` is a scalar tensor and `list_of_hyperparameters` is a list
                            of tensorflow variables that represents the hyperparameters
        :param state_history: (default: empty list) state history manager:
                                should implement methods `clear`, `append`, `__getitem__`
        :param global_step: optional instance of GlobalStep class
        """
        assert isinstance(optimizer, Optimizer)

        self.w = optimizer.w  # might be variable or MergedVariable
        #  TODO check if it works also with w as simple Variable
        self.w_t = MergedVariable.get_tensor(self.w)  # this is always a tensor

        self.tr_dynamics = optimizer.dynamics
        assert isinstance(hyper_dict, dict), '%s not allowed type. Should be a dict of' \
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

                    # equation (8) why the gradient involves alpha (p)?
                    # tf.gradients(dot( tf.gradients(ve, self.w_t), self.tr_dynamics), state_components)
                    self.p_dynamics = {ve: tf.concat(tf.gradients(lagrangian, state_components), 0)
                                       for ve, lagrangian in self.lagrangians_dict.items()}
                else:
                    # equation (8) why the gradient involves alpha (p)?
                    # tf.gradients(dot( tf.gradients(ve, self.w_t), self.tr_dynamics), self.w_t)
                    self.p_dynamics = {ve: tf.gradients(lagrangian, self.w_t)[0]
                                       for ve, lagrangian in self.lagrangians_dict.items()}  # equation (7)

                self._bk_ops = [self.p_dict[ve].assign(self.p_dynamics[ve])
                                for ve in self.val_error_dict]  # TODO add here when hp are sequ.

            with tf.name_scope('w_history_ops'):
                self._w_placeholder = tf.placeholder(self.w_t.dtype)

                self.back_hist_op = self.w.assign(self._w_placeholder)

            with tf.name_scope('hyper_derivatives'):
                # equation (10) without summation... why the gradient involves alpha (p) -> look below for expansion
                # tf.gradients(dot( tf.gradients(ve, self.w_t)), self.tr_dynamics), lambda)
                self.hyper_derivatives = [
                    (self.val_error_dict[ve], tf.gradients(lagrangian, self.val_error_dict[ve])) for ve, lagrangian in
                    self.lagrangians_dict.items()
                    ]  # list of couples (hyper_list, list of symbolic hyper_gradients)  (lists are unhashable!)

    def forward(self, T, feed_dict_supplier=None, summary_utils=None):
        if not feed_dict_supplier:
            # noinspection PyUnusedLocal
            def feed_dict_supplier(step=None): return None

        var_init = self.w.var_list(Vl_Mode.BASE) if isinstance(self.w, MergedVariable) else [self.w]
        tf.variables_initializer(var_init + [self.global_step.var]).run()

        ss = tf.get_default_session()
        self.w_hist.clear()

        for _ in range(T):
            self.w_hist.append(self.w_t.eval())
            ss.run([self.w_t, self._fw_ops, self.global_step.increase], feed_dict=feed_dict_supplier(_))
            if summary_utils: summary_utils.run(ss, _)

    def backward(self, T, validation_suppliers=None, training_supplier=None, summary_utils=None, check_if_zero=False):
        """

        :param check_if_zero:
        :param T:
        :param validation_suppliers: either a function or a dictionary {validation_error tensor: function, supplier}
        :param training_supplier:
        :param summary_utils:
        :return:
        """

        if not training_supplier:
            # noinspection PyUnusedLocal
            def training_supplier(step=None): return None
        if not validation_suppliers:  # FIXME probably won't work with the current settings.
            def validation_suppliers(): return None
        else:
            if not isinstance(validation_suppliers, dict) and len(self.val_error_dict.keys()) == 1:
                # cast validation supplier into a dict
                validation_suppliers = {list(self.val_error_dict.keys())[0]: validation_suppliers}

        # compute alpha_T using the validation set
        [tf.variables_initializer([self.p_dict[ve]]).run(feed_dict=data_supplier())
         for ve, data_supplier in validation_suppliers.items()]

        # set hyper-derivatives to 0
        hyper_derivatives = self._initialize_hyper_derivatives_res()  # TODO deal better with the hyper-derivatives
        ss = tf.get_default_session()

        if summary_utils: summary_utils.run(ss, T)

        for _ in range(T - 1, -1, -1):

            # revert w_t to w_(t-1), why not using _ -1?
            ss.run(self.back_hist_op,
                   feed_dict={self._w_placeholder: self.w_hist[self.global_step.eval() - 1]}
                   )

            fds = training_supplier(_)
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

        return hyper_derivatives

    def _initialize_hyper_derivatives_res(self):
        return {hyper: [] for hyper in self.hyper_list}

    def run_all(self, T, training_supplier=None, validation_suppliers=None, forward_su=None, backward_su=None,
                after_forward_su=None, check_if_zero=False):
        """
        Performs both forward and backward step.

        :param check_if_zero:
        :param T:
        :param training_supplier:
        :param validation_suppliers:
        :param forward_su:
        :param backward_su:
        :param after_forward_su:
        :return: A list of list: Point-wise gradients (Numpy arrays)
                of the training dynamics w.r.t. the hyper-parameters

        """
        self.forward(T, feed_dict_supplier=training_supplier, summary_utils=forward_su)
        if after_forward_su:
            after_forward_su.run(tf.get_default_session(), T)
        return self.backward(
            T, validation_suppliers=validation_suppliers,
            training_supplier=training_supplier, summary_utils=backward_su,
            check_if_zero=check_if_zero
        )

    @classmethod
    def std_collect_hyper_gradients(cls, row_gradients):
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

        self.w = optimizer.w  # might be variable or MergedVariable (never tested on Variables actually) ...
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

    def get_doh_hyper_dict(self):
        """

        :return: A dictionary of (validation errors, (list of) hyper-parameters)
                 suitable as input for Doh initializer.
        """
        return {k: [e[0] for e in v] for k, v in self.hyper_dict.items()}

    def _create_z(self, hyper):
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
        var_init = self.w.var_list(Vl_Mode.BASE) if isinstance(self.w, MergedVariable) else [self.w]
        tf.variables_initializer(var_init + self.hyper_gradient_vars + [self.global_step.var]).run()
        [tf.variables_initializer(z.components).run() for z in self.zs]

    def step_forward(self, feed_dict_supplier=None, summary_utils=None):
        if not feed_dict_supplier:
            # noinspection PyUnusedLocal
            def feed_dict_supplier(step=None): return None

        ss = tf.get_default_session()

        fd = feed_dict_supplier(self.global_step.eval())

        ss.run(self.zs_assigns, feed_dict=fd)
        ss.run(self.fw_ops, feed_dict=fd)
        if summary_utils: summary_utils.run(ss, self.global_step.eval())
        self.global_step.increase.eval()

    def hyper_gradients(self, validation_suppliers=None, new_mode=False):
        """
        Method that computes the hyper-gradient.

        :param new_mode:
        :param validation_suppliers: single  supplier or list of suppliers for the examples in the validation set

        :return: Dictionary: {hyper-parameter: hyper-gradient} or None in new_mode
        """
        if not isinstance(validation_suppliers, dict) and len(self.hyper_list) == 1:  # fine if there is only 1 hyper
            validation_suppliers = {self.hyper_list[0]: validation_suppliers}

        val_sup_lst = []
        for hyp in self.hyper_list:  # find the right validation error for hyp!
            for k, v in self.hyper_dict.items():
                all_hypers = [e[0] for e in v]
                if hyp in all_hypers:
                    val_sup_lst.append(validation_suppliers[k])
                    break

        if new_mode:
            [assign_op.eval(feed_dict=vsl(self.global_step.eval())) for
             assign_op, vsl in zip(self._hyper_assign_ops, val_sup_lst)]

        else:
            print("WARNING: things with hyper-gradients have been changed. Now they are treated as variables that "
                  "is much easier.")
            computations = [gve.eval(feed_dict=vsl(self.global_step.eval())) for
                            gve, vsl in zip(self.grad_wrt_hypers, val_sup_lst)]  # computes the actual gradients

            def cast_to_scalar_if_needed(grad):
                sh_grad = list(np.shape(grad))
                return grad[0] if len(sh_grad) == 1 and sh_grad[0] == 1 else grad

            return {hyp: cast_to_scalar_if_needed(gdd) for hyp, gdd in zip(self.hyper_list, computations)}


class ZMergedMatrix:

    def __init__(self, matrix_list):

        self.components = as_list(matrix_list)

        # assumes that you want matrices and not vectors. This means that eventual vectors are casted to matrices
        # of dimension (n, 1)
        for i, c in enumerate(self.components):
            if len(c.get_shape().as_list()) == 1:
                self.components[i] = tf.transpose(tf.stack([c]))

        self.tensor = tf.concat(self.components, 0)

    def assign(self, value_list):
        if isinstance(value_list, ZMergedMatrix):
            value_list = value_list.components
        assert len(value_list) == len(self.components), 'the length of value_list and of z, components must coincide'
        return [c.assign(v) for c, v in zip(self.components, value_list)]

    # noinspection PyUnusedLocal
    def var_list(self, mode=Vl_Mode.RAW):
        # if mode == Vl_Mode.RAW or mode == Vl_Mode.TENSOR:
        return self.components

    def __add__(self, other):
        assert isinstance(other, ZMergedMatrix)  # TODO make it a little bit more flexible (e.g. case of GD)
        assert len(other.components) == len(self.components)
        return ZMergedMatrix([c + v for c, v in zip(self.components, other.components)])


class RealTimeHO:

    def __init__(self, direct_doh, hyper_opt_dicts, hyper_projections=None, hyper_step=None):
        self.direct_doh = direct_doh

        assert isinstance(hyper_opt_dicts, (list, Optimizer)), "hyper_opt_dicts should be a single OptDict or a " \
                                                               "list of OptDict. Instead is %s" % hyper_opt_dicts
        self.hyper_opt_dicts = as_list(hyper_opt_dicts)

        self.hyper_projections = hyper_projections or []

        self.hyper_step = hyper_step or GlobalStep()

        # self.collected_hyper_gradients = {}

    def initialize(self):
        tf.variables_initializer(self.direct_doh.hyper_list + [self.hyper_step.var]).run()
        [ahd.support_variables_initializer().run() for ahd in self.hyper_opt_dicts]
        self.direct_doh.initialize()

    def hyper_batch(self, size, training_supplier=None, validation_suppliers=None, saver=None, collect_data=True,
                    apply_hyper_gradients=True):
        ss = tf.get_default_session()

        for k in range(size):
            self.direct_doh.step_forward(feed_dict_supplier=training_supplier)

        self.direct_doh.hyper_gradients(validation_suppliers=validation_suppliers, new_mode=True)  # compute hyper_grads

        if apply_hyper_gradients:
            [ss.run(hod.assign_ops) for hod in self.hyper_opt_dicts]
            [ss.run(prj) for prj in self.hyper_projections]

        self.hyper_step.increase.eval()

        if saver: saver(ss, self.hyper_step.eval(), collect_data=collect_data)  # saver should become a class...


def hyper_opt_dict(direct_doh, dynamics, **dynamics_kw_args):
    """
    Helper for building descent procedure for the hyper-parameters.

    :param direct_doh: instance of DirectDoh class
    :param dynamics:
    :param dynamics_kw_args:
    :return:
    """
    assert callable(dynamics), '%s should be an optimization dynamics generator' % dynamics
    # TODO make changes in Doh so that it follows the same structure as DirectDoh
    assert isinstance(direct_doh, ForwardHyperGradient), \
        'This should ideally work also for Doh, but is not implemented yet..'
    return [dynamics(hyp, **dynamics_kw_args, grad=hg, w_is_state=False)
            for hyp, hg in zip(direct_doh.hyper_list, direct_doh.hyper_gradient_vars)]


def positivity(hyper_list):
    return [hyp.assign(tf.maximum(hyp, tf.zeros_like(hyp))) for hyp in hyper_list]


def print_hyper_gradients(hyper_gradient_dict):
    for k, v in hyper_gradient_dict.items():
        print(k.name, v)
