import tensorflow as tf

from rfho.utils import hvp, MergedVariable, Vl_Mode, GlobalStep, ZMergedMatrix, simple_name


class Optimizer:  # Gradient descent-like optimizer
    """
    Optimizers compatible with `HyperGradient` classes.
    """

    def __init__(self, raw_w, w, assign_ops, dynamics, jac_z, learning_rate, gradient, loss):
        self.raw_w = raw_w
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

    def support_variables_initializer(self):
        """
        Returns an initialization op for the support variables (like velocity for momentum)

        :return:
        """
        return tf.variables_initializer(self.get_support_variables())

    def get_support_variables(self):
        return []

    def get_optimization_hyperparameters(self):
        return [self.learning_rate]

    def increase_global_step(self):
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

    # noinspection PyUnusedLocal
    @staticmethod
    def create(w, lr, loss=None, grad=None, w_is_state=True, name='GradientDescent'):
        """
        Just gradient descent dynamics.
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
                grad = tf.gradients(loss, MergedVariable.get_tensor(w))[0]
            dynamics = MergedVariable.get_tensor(w) - lr * grad
            if loss is not None:
                # TODO add type checking for w (should work only with vectors...)
                integral = tf.reduce_sum(MergedVariable.get_tensor(w) ** 2) / 2. - lr * loss

                def jac_z(z):
                    return ZMergedMatrix(hvp(integral, MergedVariable.get_tensor(w), z.tensor))
            else:
                jac_z = None

            return GradientDescentOptimizer(raw_w=w, w=MergedVariable.get_tensor(w),
                             assign_ops=[w.assign(dynamics)],  # TODO complete here...
                             dynamics=dynamics,
                             jac_z=jac_z,
                             gradient=grad,
                             learning_rate=lr, loss=loss)

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def get_augmentation_multiplier():
        return 0


class MomentumOptimizer(Optimizer):
    def __init__(self, raw_w, w, m, assign_ops, dynamics, jac_z, gradient, learning_rate, momentum_factor, loss):
        super(MomentumOptimizer, self).__init__(raw_w=raw_w, w=w, assign_ops=assign_ops, dynamics=dynamics, jac_z=jac_z,
                                                learning_rate=learning_rate, gradient=gradient, loss=loss)
        self.m = m
        self.momentum_factor = momentum_factor

        self.algorithmic_hypers[self.momentum_factor] = self.d_dynamics_d_momentum_factor

    def get_support_variables(self):
        return [self.m]

    def get_optimization_hyperparameters(self):
        return super().get_optimization_hyperparameters() + [self.momentum_factor]

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

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def get_augmentation_multiplier():
        return 1

    @staticmethod
    def create(w, lr, mu, loss=None, grad=None, w_is_state=True, name='Momentum'):
        """


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
                assert len(w.var_list(Vl_Mode.TENSOR)) == 2, "%s is not augmented correctly, len of w.var_list(" \
                                                             "Vl_Mode.TENSOR should be 2, but is " \
                                                             "%d" % (w, len(w.var_list(Vl_Mode.TENSOR)))

                w_base, m = w.var_list(Vl_Mode.TENSOR)
            else:
                w_base = w
                m = tf.Variable(tf.zeros(w.get_shape()))
            if grad is None:
                grad = tf.gradients(loss, w_base)[0]

            w_base_k = w_base - lr * (mu * m + grad)  # * (mu * m + (1. - mu) * grad)   old
            m_k = mu * m + grad  # * (1. - mu)

            def jac_z(z):
                r, u = z.var_list(Vl_Mode.TENSOR)

                assert loss is not None, 'Should specify loss to use jac_z'

                hessian_r_product = hvp(loss=loss, w=w_base, v=r)

                print('hessian_r_product', hessian_r_product)

                res = [
                    r - lr * mu * u - lr * hessian_r_product,
                    hessian_r_product + mu * u
                ]

                print('res', res)

                return ZMergedMatrix(res)

            dynamics = tf.concat([w_base_k, m_k], 0) if w_base_k.get_shape().ndims != 0 \
                else tf.stack([w_base_k, m_k], 0)  # for scalar w

            if w_is_state:
                w_base_mv, m_mv = w.var_list(Vl_Mode.RAW)
            else:
                w_base_mv, m_mv = w_base, m

            return MomentumOptimizer(
                w=w_base,
                m=m,
                assign_ops=[w_base_mv.assign(w_base_k), m_mv.assign(m_k)],
                dynamics=dynamics,
                jac_z=jac_z, gradient=grad, learning_rate=lr, momentum_factor=mu, raw_w=w,
                loss=loss
            )


class AdamOptimizer(MomentumOptimizer):
    def __init__(self, raw_w, w, m, v, assign_ops, global_step, dynamics, jac_z, gradient, learning_rate,
                 momentum_factor, second_momentum_factor, loss):
        super().__init__(w=w, m=m, assign_ops=assign_ops, dynamics=dynamics, jac_z=jac_z, gradient=gradient,
                         learning_rate=learning_rate, momentum_factor=momentum_factor, raw_w=raw_w, loss=loss)
        self.v = v
        self.global_step = global_step
        self.second_momentum_factor = second_momentum_factor

        self.algorithmic_hypers[self.second_momentum_factor] = self.d_dynamics_d_second_momentum_factor

    def support_variables_initializer(self):
        return tf.variables_initializer([self.m, self.v, self.global_step.var])

    def increase_global_step(self):
        self.global_step.increase.eval()

    def get_optimization_hyperparameters(self):
        return super().get_optimization_hyperparameters() + [self.second_momentum_factor]  # could be also epsilon?

    def d_dynamics_d_momentum_factor(self):
        raise NotImplementedError()  # TODO

    def d_dynamics_d_learning_rate(self):
        raise NotImplementedError()  # TODO

    def d_dynamics_d_second_momentum_factor(self):
        raise NotImplementedError()  # TODO

    def d_dynamics_d_hyper_loss(self, grad_loss_term):
        raise NotImplementedError()  # TODO

    # noinspection PyMissingOrEmptyDocstring
    @staticmethod
    def get_augmentation_multiplier():
        return 2

    @staticmethod
    def create(w, lr=1.e-3, beta1=.9, beta2=.999, eps=1.e-8, global_step=None,
               loss=None, grad=None, w_is_state=True, name='Adam'):  # FIXME rewrite this
        """
        Adam optimizer.
        :param w:
        :param lr:
        :param beta1:
        :param beta2:
        :param eps:
        :param global_step:
        :param loss:
        :param grad:
        :param w_is_state:
        :param name:
        :return:
        """
        # beta1_pow = tf.Variable(beta1)  # for the moment skip the implementation of this optimization.
        assert grad is not None or loss is not None, "One between grad or loss must be given"
        with tf.name_scope(name):
            if w_is_state:

                assert isinstance(w, MergedVariable), "%s is not instance of MergedVariable" % w
                assert len(w.var_list(Vl_Mode.TENSOR)) == 3, "%s is not augmented correctly, len of w.var_list(" \
                                                             "Vl_Mode.TENSOR should be 3, but is " \
                                                             "%d" % (w, len(w.var_list(Vl_Mode.TENSOR)))

                w_base, m, v = w.var_list(Vl_Mode.TENSOR)
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

            lr_k = lr * tf.sqrt(1. - tf.pow(beta2, tf.to_float(global_step.var + 1))) / (
                1. - tf.pow(beta1, tf.to_float(global_step.var + 1)))
            w_base_k = w_base - lr_k * (beta1 * m + (1. - beta1) * grad) / tf.sqrt(
                beta2 * v + (1. - beta2) * grad ** 2 + eps)

            jac_z = None  # TODO!!!!!

            # noinspection PyUnresolvedReferences
            dynamics = tf.concat([w_base_k, m_k, v_k], 0) if w_base_k.get_shape().ndims != 0 \
                else tf.stack([w_base_k, m_k, v_k], 0)  # scalar case

            if w_is_state:
                w_base_mv, m_mv, v_mv = w.var_list(Vl_Mode.RAW)
            else:
                w_base_mv, m_mv, v_mv = w_base, m, v

            return AdamOptimizer(
                w=w_base,
                m=m, v=v, global_step=global_step,
                assign_ops=[w_base_mv.assign(w_base_k), m_mv.assign(m_k), v_mv.assign(v_k)],
                dynamics=dynamics,
                jac_z=jac_z, gradient=grad, learning_rate=lr, momentum_factor=beta1, second_momentum_factor=beta2,
                raw_w=w, loss=loss
            )
