"""
This module contains optimizers.
"""

import tensorflow as tf

from rfho.utils import hvp, MergedVariable, VlMode, GlobalStep, ZMergedMatrix, simple_name, l_diag_mul


class Optimizer:  # Gradient descent-like optimizer
    """
    Optimizers compatible with `HyperGradient` classes.
    """

    def __init__(self, raw_w, w, assign_ops, dynamics, jac_z, learning_rate, gradient, loss):
        self.raw_w = raw_w  # TODO distinction no more needed
        self.w = w
        self.assign_ops = assign_ops
        self.dynamics = dynamics
        self.jac_z = jac_z
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.loss = loss

        self.algorithmic_hypers = {
            self.learning_rate: self.d_dynamics_d_learning_rate
        }

    def get_assign_op(self):
        return self.raw_w.assign(self.dynamics)

    def support_variables_initializer(self):
        """
        Returns an initialization op for the support variables (like velocity for momentum)

        :return:
        """
        return tf.variables_initializer(self.get_support_variables())

    def get_support_variables(self):
        """

        :return: support variables for this optimizers (like velocity for momentum)
        """
        return []

    def get_optimization_hyperparameters(self, only_variables=False):
        """
        Returns the optimizer hyperparameters.

        :param only_variables: if true returns only the hyperparameters that are also `tf.Variable` (which can
                                be optimized)

        :return:
        """
        return Optimizer._filter_variables([self.learning_rate], only_variables)

    def get_natural_hyperparameter_constraints(self):
        """

        :return: a list of ops which represent the projections of algorithmic hyperparameter onto their
                 natural domain (e.g. learning rate >= 0, momentum factor in [0, 1)...)
        """
        hypers = self.get_optimization_hyperparameters(only_variables=True)
        constraints = []
        if self.learning_rate in hypers:
            constraints.append(self.learning_rate.assign(tf.maximum(self.learning_rate, 0.)))
        return constraints

    @staticmethod
    def _filter_variables(vr_list, only_variables):
        return [v for v in vr_list if isinstance(v, tf.Variable)] if only_variables else vr_list

    def increase_global_step(self):
        """
        If there is a global step, increases it

        :return:
        """
        pass

    def d_dynamics_d_learning_rate(self):
        """

        :return: Partial derivative of dynamics w.r.t. learning rate
        """
        return ZMergedMatrix(-self.gradient)

    def d_dynamics_d_hyper_loss(self, grad_loss_term):
        """
        Helper function for building the partial derivative of the dynamics w.r.t. an hyperparameter
        inside the loss function, given the gradient or Jacobian of loss w.r.t.

        :param grad_loss_term: should be \nabla R
        :return: Partial derivative of dynamics w.r.t. weighting hyperparameter (e.g. gamma)
        """
        return ZMergedMatrix(-self.learning_rate * grad_loss_term)

    def _auto_d_dynamics_d_hyper_no_zmm(self, hyper):
        d_loss_d_lambda = tf.gradients(self.loss, hyper)[0]
        assert d_loss_d_lambda is not None, "No gradient of tensor %s w.r.t hyperparameter %s" \
                                            % (simple_name(self.loss), simple_name(hyper))

        shape = d_loss_d_lambda.get_shape()
        if shape.ndims == 1:  # hyper is a vector
            return tf.stack([tf.gradients(d_loss_d_lambda[i], self.w)[0]
                             for i in range(shape[0].value)], axis=1)

        elif shape.ndims == 0:
            return tf.gradients(d_loss_d_lambda, self.w)[0]
        else:
            raise NotImplementedError('For forward mode hyperparameters should be either scalar or vectors. \n'
                                      '%s is a %d rank tensor instead' % (simple_name(hyper), shape.ndims))

    def auto_d_dynamics_d_hyper(self, hyper):
        """
        Automatic attempt to build the term d(phi)/(d hyper)
        
        :param hyper: 
        :return: 
        """
        if hyper in self.algorithmic_hypers:
            return self.algorithmic_hypers[hyper]()  # the call to the function is here because otherwise it could
        # rise an error at the initialization part in the case that the optimizer is used in gradient mode
        # (i.e. loss is not provided in creation method)
        return self.d_dynamics_d_hyper_loss(self._auto_d_dynamics_d_hyper_no_zmm(hyper))

    @staticmethod
    def get_augmentation_multiplier():
        """
        Convenience method for `augment` param of `vectorize_model`

        :return: Returns dim(state)/dim(parameter vector) for this optimizer.
        """
        return 0


class GradientDescentOptimizer(Optimizer):
    """
    Class for gradient descent to be used with HyperGradients
    """

    # noinspection PyUnusedLocal
    @staticmethod
    def create(w, lr, loss=None, grad=None, w_is_state=True, name='GradientDescent'):
        """
        Just gradient descent.
        :param w_is_state: Just for compatibility with other create methods.
        Gradient descent does not require auxiliary variables!
        :param loss: (optional) scalar loss
        :param w: must be a single variable or a tensor (use models.vectorize_model). No lists here!
        :param lr:
        :param name:
        :param grad: (optional) gradient Tensor
        :return: tf.Tensor for gradient descent dynamics
        """
        assert grad is not None or loss is not None, "One between grad or loss must be given"
        with tf.name_scope(name):
            if grad is None:
                grad = tf.gradients(loss,
                                    # MergedVariable.get_tensor(w),
                                    w)[0]
            dynamics = w - lr * grad
            if loss is not None:
                # TODO add type checking for w (should work only with vectors...)
                integral = tf.reduce_sum(
                    w** 2) / 2. - lr * loss

                def _jac_z(z):
                    return ZMergedMatrix(hvp(integral, w,
                                             # MergedVariable.get_tensor(w),
                                             z.tensor))
            else:
                _jac_z = None

            return GradientDescentOptimizer(raw_w=w, w=w,
                                            assign_ops=[w.assign(dynamics)],  # TODO complete here...
                                            dynamics=dynamics,
                                            jac_z=_jac_z,
                                            gradient=grad,
                                            learning_rate=lr, loss=loss)

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def get_augmentation_multiplier():
        return 0


# noinspection PyMissingOrEmptyDocstring
class MomentumOptimizer(Optimizer):
    def __init__(self, raw_w, w, m, assign_ops, dynamics, jac_z, gradient, learning_rate, momentum_factor, loss):
        super(MomentumOptimizer, self).__init__(raw_w=raw_w, w=w, assign_ops=assign_ops, dynamics=dynamics, jac_z=jac_z,
                                                learning_rate=learning_rate, gradient=gradient, loss=loss)
        self.m = m
        self.momentum_factor = momentum_factor

        self.algorithmic_hypers[self.momentum_factor] = self.d_dynamics_d_momentum_factor

    def get_support_variables(self):
        return [self.m]

    def get_optimization_hyperparameters(self, only_variables=False):
        return Optimizer._filter_variables(
            super().get_optimization_hyperparameters() + [self.momentum_factor], only_variables)

    def d_dynamics_d_learning_rate(self):
        return ZMergedMatrix([- self.momentum_factor * self.m - self.gradient,
                              tf.zeros(self.m.get_shape())
                              ])

    def d_dynamics_d_momentum_factor(self):
        return ZMergedMatrix([- (self.learning_rate * self.m), self.m])

    def d_dynamics_d_hyper_loss(self, grad_loss_term):
        return ZMergedMatrix([
            - self.learning_rate * grad_loss_term,
            grad_loss_term
        ])

    @staticmethod
    def get_augmentation_multiplier():
        return 1

    def get_natural_hyperparameter_constraints(self):
        hypers = self.get_optimization_hyperparameters(only_variables=True)
        constraints = super().get_natural_hyperparameter_constraints()
        if self.momentum_factor in hypers:
            constraints.append(self.momentum_factor.assign(
                tf.minimum(tf.maximum(self.momentum_factor, 0.), 0.9999)))
        return constraints

    @staticmethod
    def create(w, lr, mu=.9, loss=None, grad=None, w_is_state=True, name='Momentum', _debug_jac_z=False):
        """
        Constructor for momentum optimizer

        :param _debug_jac_z: 
        :param mu:
        :param w:
        :param lr:
        :param loss:
        :param grad:
        :param w_is_state:
        :param name:
        :return: a new MomentumOptimizer object
        """
        # beta1_pow = tf.Variable(beta1)  # for the moment skip the implementation of this optimization.
        assert grad is not None or loss is not None, "One between grad or loss must be given"
        with tf.name_scope(name):
            if w_is_state:

                assert isinstance(w, MergedVariable), "%s is not instance of MergedVariable" % w
                assert len(w.var_list(VlMode.TENSOR)) == 2, "%s is not augmented correctly, len of w.var_list(" \
                                                            "VlMode.TENSOR should be 2, but is " \
                                                            "%d" % (w, len(w.var_list(VlMode.TENSOR)))

                w_base, m = w.var_list(VlMode.TENSOR)
            else:
                w_base = w
                m = tf.Variable(tf.zeros(w.get_shape()))
            if grad is None:
                grad = tf.gradients(loss, w_base)[0]

            w_base_k = w_base - lr * (mu * m + grad)  # * (mu * m + (1. - mu) * grad)   old
            m_k = mu * m + grad  # * (1. - mu)

            # noinspection PyUnresolvedReferences
            dynamics = tf.concat([w_base_k, m_k], 0) if w_base_k.get_shape().ndims != 0 \
                else tf.stack([w_base_k, m_k], 0)  # for scalar w

            # noinspection PyUnresolvedReferences
            def _jac_z(z):
                if _debug_jac_z:  # I guess this would take an incredible long time to compile for large systems
                    d = dynamics.get_shape().as_list()[0]
                    d2 = d // 2
                    jac_1_1 = tf.stack([
                        tf.gradients(w_base_k[i], w_base)[0] for i in range(d2)
                    ])
                    jac_2_1 = tf.stack([
                        tf.gradients(m_k[i], w_base)[0] for i in range(d2)
                    ])
                    # jac_1 = tf.concat([jac_1_1, jac_2_1], axis=0)

                    jac_1_2 = tf.stack([
                        tf.gradients(w_base_k[i], m)[0] for i in range(d2)
                    ])
                    jac_2_2 = tf.stack([
                        tf.gradients(m_k[i], m)[0] for i in range(d2)
                    ])
                    # jac_2 = tf.concat([jac_1_2, jac_2_2], axis=0)

                    # jac = tf.concat([jac_1, jac_2], axis=1, name='Jacobian')

                    # mul = tf.matmul(jac, z.tensor)
                    #
                    # return ZMergedMatrix([
                    #     mul[:d2, :],
                    #     mul[d2, :]
                    # ])
                    r, u = z.var_list(VlMode.TENSOR)
                    return ZMergedMatrix([
                        tf.matmul(jac_1_1, r) + tf.matmul(jac_1_2, u),
                        tf.matmul(jac_2_1, r) + tf.matmul(jac_2_2, u)
                    ])
                else:
                    r, u = z.var_list(VlMode.TENSOR)

                    assert loss is not None, 'Should specify loss to use jac_z'

                    hessian_r_product = hvp(loss=loss, w=w_base, v=r)

                    # print('hessian_r_product', hessian_r_product)

                    res = [
                        r - lr * mu * u - lr * hessian_r_product,
                        hessian_r_product + mu * u
                    ]

                    return ZMergedMatrix(res)

            if w_is_state:
                w_base_mv, m_mv = w.var_list(VlMode.RAW)
            else:
                w_base_mv, m_mv = w_base, m

            return MomentumOptimizer(
                w=w_base,
                m=m,
                assign_ops=[w_base_mv.assign(w_base_k), m_mv.assign(m_k)],
                dynamics=dynamics,
                jac_z=_jac_z, gradient=grad, learning_rate=lr, momentum_factor=mu, raw_w=w,
                loss=loss
            )


# noinspection PyMissingOrEmptyDocstring
class AdamOptimizer(MomentumOptimizer):
    """
    Class for adam optimizer 
    """

    def __init__(self, raw_w, w, m, v, assign_ops, global_step, dynamics, jac_z, gradient, learning_rate,
                 momentum_factor, second_momentum_factor, loss, d_dyn_d_lr, d_dyn_d_hyper):
        super().__init__(w=w, m=m, assign_ops=assign_ops, dynamics=dynamics, jac_z=jac_z, gradient=gradient,
                         learning_rate=learning_rate, momentum_factor=momentum_factor, raw_w=raw_w, loss=loss)
        self.v = v
        self.global_step = global_step
        self.second_momentum_factor = second_momentum_factor

        self._d_dyn_d_lr = d_dyn_d_lr
        self._d_dyn_d_hyper = d_dyn_d_hyper

        self.algorithmic_hypers[self.second_momentum_factor] = self.d_dynamics_d_second_momentum_factor

    def support_variables_initializer(self):
        return tf.variables_initializer([self.m, self.v, self.global_step.var])

    def increase_global_step(self):
        self.global_step.increase.eval()

    def get_optimization_hyperparameters(self, only_variables=False):
        return Optimizer._filter_variables(
            super().get_optimization_hyperparameters() + [self.second_momentum_factor],  # could be also epsilon?
            only_variables
        )

    def get_natural_hyperparameter_constraints(self):
        hypers = self.get_optimization_hyperparameters(only_variables=True)
        constraints = super().get_natural_hyperparameter_constraints()
        if self.second_momentum_factor in hypers:
            constraints.append(self.second_momentum_factor.assign(
                tf.minimum(tf.maximum(self.second_momentum_factor, 0.), 0.9999)))
        return constraints

    def d_dynamics_d_momentum_factor(self):
        raise NotImplementedError()  # TODO

    def d_dynamics_d_learning_rate(self):
        return self._d_dyn_d_lr()

    def d_dynamics_d_second_momentum_factor(self):
        raise NotImplementedError()  # TODO

    def d_dynamics_d_hyper_loss(self, grad_loss_term):
        return self._d_dyn_d_hyper(grad_loss_term)

    @staticmethod
    def get_augmentation_multiplier():
        return 2

    @staticmethod
    def create(w, lr=1.e-3, beta1=.9, beta2=.999, eps=1.e-6, global_step=None,
               loss=None, grad=None, w_is_state=True, name='Adam',
               _debug_jac_z=False):  # FIXME rewrite this
        """
        Adam optimizer.
        
        :param w: all weight vector
        :param lr: learning rate
        :param beta1: first momentum factor
        :param beta2: second momentum factor
        :param eps: term for numerical stability (higher than proposed default)
        :param global_step: 
        :param loss: scalar tensor 
        :param grad: vector tensor
        :param w_is_state:
        :param name:
        :param _debug_jac_z: 
        :return:
        """
        # beta1_pow = tf.Variable(beta1)  # for the moment skip the implementation of this optimization.
        assert grad is not None or loss is not None, "One between grad or loss must be given"
        with tf.name_scope(name):
            if w_is_state:

                assert isinstance(w, MergedVariable), "%s is not instance of MergedVariable" % w
                assert len(w.var_list(VlMode.TENSOR)) == 3, "%s is not augmented correctly, len of w.var_list(" \
                                                            "VlMode.TENSOR should be 3, but is " \
                                                            "%d" % (w, len(w.var_list(VlMode.TENSOR)))

                w_base, m, v = w.var_list(VlMode.TENSOR)
            else:
                w_base = w
                m = tf.Variable(tf.zeros(w.get_shape()))
                v = tf.Variable(tf.zeros(w.get_shape()))
            if grad is None:
                grad = tf.gradients(loss, w_base)[0]
            if global_step is None:
                global_step = GlobalStep()

            m_k = beta1 * m + (1. - beta1) * grad
            v_k = beta2 * v + (1. - beta2) * grad ** 2

            bias_correction = tf.sqrt(1. - tf.pow(beta2, tf.to_float(global_step.var + 1))) / (
                1. - tf.pow(beta1, tf.to_float(global_step.var + 1)))
            lr_k = lr * bias_correction

            v_epsilon_k = beta2 * v + (1. - beta2) * grad ** 2 + eps
            v_tilde_k = tf.sqrt(v_epsilon_k)  # + eps
            """
            to make it the same as tensorflow adam optimizer the eps should go after the square root... this
            brings however some problems in the computation of the hypergradient, therefore we put it inside!
            SHOULD BETTER INVESTIGATE THE ISSUE. (maybe the jacobian computation should be done again)
            """

            # TODO THESE QUANTITIES ARE NEEDED FOR FORWARD-HG IN VARIOUS PLACES... FIND A BETTER WAY TO COMPUTE THEM
            # ONLY IF NEEDED
            v_k_eps_32 = tf.pow(v_epsilon_k, 1.5)
            pre_j_11_out = - lr_k * (
                (1. - beta1) / v_tilde_k - ((1. - beta2) * grad * m_k) / v_k_eps_32
            )
            pre_j_31_out = 2. * (1. - beta2) * grad

            w_base_k = w_base - lr_k * (beta1 * m + (1. - beta1) * grad) / v_tilde_k

            # noinspection PyUnresolvedReferences
            def _jac_z(z):
                if _debug_jac_z:  # I guess this would take an incredible long time to compile for large systems
                    d = dynamics.get_shape().as_list()[0] // 3
                    r, u, s = z.var_list(VlMode.TENSOR)

                    j11 = tf.stack([
                        tf.gradients(w_base_k[i], w_base)[0] for i in range(d)
                    ])
                    j12 = tf.stack([
                        tf.gradients(w_base_k[i], m)[0] for i in range(d)
                    ])
                    j13 = tf.stack([
                        tf.gradients(w_base_k[i], v)[0] for i in range(d)
                    ])
                    j1 = tf.concat([j11, j12, j13], axis=1)
                    jz1 = tf.matmul(j11, r) + tf.matmul(j12, u) + tf.matmul(j13, s)

                    # second block
                    j21 = tf.stack([
                        tf.gradients(m_k[i], w_base)[0] for i in range(d)
                    ])
                    j22 = tf.stack([
                        tf.gradients(m_k[i], m)[0] for i in range(d)
                    ])
                    j23 = tf.stack([
                        tf.gradients(m_k[i], v)[0] for i in range(d)
                    ])
                    j2 = tf.concat([j21, j22, j23], axis=1)
                    jz2 = tf.matmul(j21, r) + tf.matmul(j22, u) + tf.matmul(j23, s)

                    # third block
                    j31 = tf.stack([
                        tf.gradients(v_k[i], w_base)[0] for i in range(d)
                    ])
                    j32 = tf.stack([
                        tf.gradients(v_k[i], m)[0] for i in range(d)
                    ])
                    j33 = tf.stack([
                        tf.gradients(v_k[i], v)[0] for i in range(d)
                    ])
                    j3 = tf.concat([j31, j32, j33], axis=1)
                    jz3 = tf.matmul(j31, r) + tf.matmul(j32, u) + tf.matmul(j33, s)

                    tf.concat([j1, j2, j3], axis=0, name='Jacobian')

                    return ZMergedMatrix([jz1, jz2, jz3])

                else:
                    assert loss is not None, 'Should specify loss to use jac_z'

                    r, u, s = z.var_list(VlMode.TENSOR)

                    with tf.name_scope('Jac_Z'):

                        hessian_r_product = hvp(loss=loss, w=w_base, v=r, name='hessian_r_product')
                        # hessian_r_product = hvp(loss=loss, w=w.tensor, v=z.tensor, name='hessian_r_product')[:d, :d]

                        j_11_r_tilde = l_diag_mul(pre_j_11_out, hessian_r_product, name='j_11_r_tilde')
                        j_11_r = tf.identity(j_11_r_tilde + r, 'j_11_r')

                        j_12_u_hat = tf.identity(- lr_k * beta1 / v_tilde_k, name='j_12_u_hat')
                        j_12_u = l_diag_mul(j_12_u_hat, u, name='j_12_u')

                        j_13_s_hat = tf.identity(lr_k * beta2 * m_k / (2 * v_k_eps_32), name='j_13_s_hat')
                        j_13_s = l_diag_mul(j_13_s_hat, s, name='j_13_s')

                        jac_z_1 = tf.identity(j_11_r + j_12_u + j_13_s, name='jac_z_1')
                        # end first bock

                        j_21_r = tf.identity((1. - beta1) * hessian_r_product, name='j_21_r')
                        j_22_u = tf.identity(beta1 * u, name='j_22_u')
                        # j_23_s = tf.zeros_like(s)  # would be...

                        jac_z_2 = tf.identity(j_21_r + j_22_u, name='jac_z_2')
                        # end second block

                        j_31_r = l_diag_mul(pre_j_31_out, hessian_r_product, name='j_31_r')
                        # j_32_u = tf.zeros_like(u)  # would be
                        j_33_s = tf.identity(beta2 * s, name='j_33_s')
                        jac_z_3 = tf.identity(j_31_r + j_33_s, name='jac_z_3')

                        res = [jac_z_1, jac_z_2, jac_z_3]
                        # print('res', res)

                        return ZMergedMatrix(res)

            # algorithmic partial derivatives (as functions so that we do not create unnecessary nodes
            def _d_dyn_d_lr():
                res = [
                    - bias_correction * m_k / v_tilde_k,
                    tf.zeros_like(m_k),
                    tf.zeros_like(v_k)  # just aesthetics
                ]
                return ZMergedMatrix(res)

            def _d_dyn_d_hyp_gl(cross_der_l):
                dwt_dl_hat = pre_j_11_out
                dwt_dl = l_diag_mul(dwt_dl_hat, cross_der_l)

                dmt_dl = (1 - beta1) * cross_der_l

                dvt_dl = l_diag_mul(pre_j_31_out, cross_der_l)
                return ZMergedMatrix([dwt_dl, dmt_dl, dvt_dl])

            # noinspection PyUnresolvedReferences
            dynamics = tf.concat([w_base_k, m_k, v_k], 0) if w_base_k.get_shape().ndims != 0 \
                else tf.stack([w_base_k, m_k, v_k], 0)  # scalar case

            if w_is_state:
                w_base_mv, m_mv, v_mv = w.var_list(VlMode.RAW)
            else:
                w_base_mv, m_mv, v_mv = w_base, m, v

            return AdamOptimizer(
                w=w_base,
                m=m, v=v, global_step=global_step,
                assign_ops=[w_base_mv.assign(w_base_k), m_mv.assign(m_k), v_mv.assign(v_k)],
                dynamics=dynamics,
                jac_z=_jac_z, gradient=grad, learning_rate=lr, momentum_factor=beta1, second_momentum_factor=beta2,
                raw_w=w, loss=loss, d_dyn_d_lr=_d_dyn_d_lr, d_dyn_d_hyper=_d_dyn_d_hyp_gl
            )
