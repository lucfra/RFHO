# import data
# import numpy as np
# working with placeholders
import contextlib
import os
from functools import reduce

import tensorflow as tf
from rfho.utils import MergedVariable
import tensorflow.contrib.graph_editor as ge
import rfho.utils as utils

test = False
do_print = False


def calc_mb(_shape, _type=32):
    from functools import reduce
    import operator
    return 1. * reduce(operator.mul, _shape, 1) * _type * 1.25e-7


def pvars(_vars, _tabs=0):
    """utility function to print the arg variables"""
    if do_print:

        print('\t' * _tabs, '-' * 10, 'START', '-' * 10)
        for k, v in _vars.items():
            print('\t' * _tabs, k, ':', v)
        print('\t' * _tabs, '-' * 10, 'END', '-' * 10)


# layers util funcs ########


def uppool(value, name='uppool'):  # TODO TBD??
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
    Note that the only dimension that can be unspecified is the first one (b)

    :param name:
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]

    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        print(value)
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            # out = tf.concat(i, [out, tf.zeros_like(out)])  #original implementation added zeros
            out = tf.concat([out, tf.identity(out)], i)  # copies values
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def convolutional_layer_2d(init_w=None, init_b=tf.zeros, strides=(1, 1, 1, 1),
                           padding='SAME', act=tf.nn.relu):
    """
    Helper function for 2d convolutional layer

    :param padding:
    :param init_w:
    :param init_b:
    :param strides:
    :param act:
    :return: an initializer
    """
    if init_w is None: init_w = lambda shape: tf.truncated_normal(shape, stddev=.1)

    def _init(_input, shape):
        _W = create_or_reuse(init_w, shape, name='W')
        _b = create_or_reuse(init_b, [shape[-1]], name='b')
        linear_activation = tf.nn.conv2d(_input, _W, strides=strides, padding=padding, name='linear_activation') + _b
        activation = act(linear_activation)
        return _W, _b, activation

    return _init


def convolutional_layer2d_maxpool(init_w=None, init_b=tf.zeros, strides=(1, 1, 1, 1),
                                  padding='SAME', act=tf.nn.relu, **maxpool_kwargs):
    init_cnv = convolutional_layer_2d(init_w, init_b, strides, padding, act)
    maxpool_kwargs.setdefault('ksize', (1, 1, 1, 1))
    maxpool_kwargs.setdefault('strides', (1, 2, 2, 1))
    maxpool_kwargs.setdefault('padding', 'SAME')

    def _init(_input, shape):
        _W, _b, activation = init_cnv(_input, shape)
        return _W, _b, tf.nn.max_pool(activation, **maxpool_kwargs)

    return _init


def convolutional_layer2d_uppool(init_w=None, init_b=tf.zeros, strides=(1, 1, 1, 1),
                                 padding='SAME', act=tf.nn.relu, **uppool_kwargs):
    init_cnv = convolutional_layer_2d(init_w, init_b, strides, padding, act)

    def _init(_input, shape):
        _W, _b, activation = init_cnv(_input, shape)
        return _W, _b, uppool(activation, **uppool_kwargs)

    return _init


def dropout_activation(_keep_prob, _activ=tf.nn.relu):
    def _int(_v, name='default_name'):
        return tf.nn.dropout(_activ(_v, name), _keep_prob)

    return _int


def create_or_reuse(init_or_variable, shape, name='var'):
    """
    Creates a variable given a shape or does nothing if `init_or_variable` is already a Variable.

    :param init_or_variable:
    :param shape:
    :param name:
    :return:
    """
    return init_or_variable if isinstance(init_or_variable, tf.Variable) \
        else tf.Variable(init_or_variable(shape), name=name)


def mixed_activation(*activations, proportions=None):
    def generate(lin_act, name='mixed_activation'):
        nonlocal proportions, activations
        if proportions:  # argument check
            sum_proportions = sum(proportions)
            assert sum_proportions <= 1, "proportions must sum up to at most 1: instead %d" % sum_proportions
            if sum_proportions < 1.: proportions += [1. - sum_proportions]
        else:
            proportions = [1 / len(activations)] * len(activations)

        N = lin_act.get_shape().as_list()[1]

        calculated_partitions = reduce(
            lambda v1, v2: v1 + [sum(v1) + v2],
            [int(N * prp) for prp in proportions],
            [0]
        )
        calculated_partitions[-1] = N

        with tf.name_scope(name):
            parts = [act(lin_act[:, d1:d2]) for act, d1, d2
                     in zip(activations, calculated_partitions, calculated_partitions[1:])]
            return tf.concat(parts, 1)

    return generate


def ffnn_layer(init_w=tf.contrib.layers.xavier_initializer(),  # OK
               init_b=tf.zeros,
               activ=tf.nn.relu, benchmark=True):
    """
    Helper for fully connected layer

    :param init_w:
    :param init_b:
    :param activ:
    :param benchmark:
    :return:
    """

    def _int(_input, _shape):
        pvars(vars(), 1)
        _W = create_or_reuse(init_w, _shape, name='W')
        _b = create_or_reuse(init_b, [_shape[1]], name='b')

        mul = utils.matmul(_input, _W, benchmark=benchmark)

        _lin_activ = mul + _b
        _activ = activ(_lin_activ, name='activation')
        return _W, _b, _activ

    return _int


# def ffnn_lin_out(init_w=tf.zeros, init_b=tf.zeros, benchmark=True):
#     return ffnn_layer(init_w, init_b, tf.identity, benchmark=benchmark)

# standard layers end ##############


def vectorize_model(model_vars, *o_outs, augment=0):
    """
    Function that "vectorizes" a model (as a computation graph).

    Given a model written in a "standard way", i.e. with parameters organized  in k-rank tensors as needed
    (matrices and vectors for linearities, 3 or 4 rank tensors for convolutional kernels etc.), returns the same model
    with all the parameters organized in a single vector.

    It is believed that having coded the models in this way, it will be much simpler to implement future algorithms,
    especially when relying on the second-order derivatives (Hessian of objective function), since in such a framework,
    the model will be entirely dependent by its single parameter vector and described by the resulting computation
    graph.

    Since the the downward process of vectorizing the model does not work, it is
     necessary to modify the computation graph accordingly to have an upward dependence
    from the all weights vector to the single parameters.

    SOLVED! { The technical caveat is that the variables must be encapsulated into an op (tf.identity) otherwise,
    for some
    unknown reason, the substitution process is not possible. I've tried in several different ways but they all failed.
    Substantially you need to keep track of the tf.identity(variable) and use the resulting tensor to build up the model
    and then also of the initial value of variable. Probably it is not necessary to keep track of  variable itself. }


    :param model_or_var_list: list of variables of the model or initializers
    :param o_outs: output_variables, list or tensor. (e.g. model output)
    :param augment: (int: default 0) augment the all weights vector by creating augumented variables (initialized at 0)
                    that mirror rank and dimensions of the variables in `model_vars`. The process is repeated
                    `augment` times.
                    This new variables can be  accessed with methods in `MergedVariable`.
                    The common usage is to prepare the model to be optimized with optimizers that require states
                    such as `MomentumOptimizer` (`augment=1`) or `AdamOptimizer` (`augment=2`).

    :return: a list which has as first element the `MergedVariable` that represents the all weights vector. Remaining
                elements are the outputs
                in the modified graph. These new outs are the same computed by the initial model
            (by the computation graph in which the model lives) but with dependencies form the all_weight_vector.

    """
    # assert len(model_vars) == len(model_vars_tensor), 'length of model_vars and model_var_tensor do not match'
    assert len(model_vars) > 0, 'no variables in model_vars!'
    outs = [tf.identity(o) if isinstance(o, tf.Variable) else o for o in o_outs]
    # TODO implement recursive call for nested lists
    with model_vars[0].graph.as_default():

        true_w = MergedVariable(model_vars)

        if augment:
            augmented_variables = [true_w]
            for k in range(augment):
                with tf.name_scope('augmented_variables'):
                    with tf.name_scope(str(k)):
                        with tf.name_scope('original_variables'):
                            tmp_augmented = [tf.Variable(tf.zeros(v.get_shape().as_list()), name=v.name.split(':')[0])
                                             for v in model_vars]
                        augmented_variables.append(MergedVariable(tmp_augmented))

            w = MergedVariable(augmented_variables)
        else:
            w = true_w

        # with open(os.devnull, 'w') as dnf:
        # with utils.suppress_stdout_stderr():  # may cause ERROR TOO MANY FILES OPENED!.:((
        new_outs = ge.graph_replace(outs, w.generate_swap_dict())

        # with utils.suppress_stdout_stderr():  # FIXME deprecation here on GraphKey usage... now redirecting outs
        #     new_outs = ge.graph_replace(outs, w.generate_swap_dict())

    return [w] + new_outs


class Network(object):
    """
    Base object for models
    """

    def __init__(self, _input, name, deterministic_initialization=False):
        """
        Creates an object that represent a network. Important attributes of a Network object are

        `var_list`: list of tf.Variables that constitute the parameters of the model

        `inp`: list, first element is `_input` and last should be output of the model. Other entries can be
        hidden layers activations.

        :param _input: tf.Tensor, input of this model.
        """
        super(Network, self).__init__()

        self.name = name

        self.deterministic_initialization = deterministic_initialization
        self._var_list_initial_values = []
        self._var_init_placeholder = None
        self._assign_int = []
        self._var_initializer_op = None

        self.Ws = []
        self.bs = []
        self.inp = [_input]
        self.out = None  # for convenience
        self.var_list = []

        self.active_gen = []
        self.active_gen_kwargs = []

        self.w = None

    def _std_collections(self):
        self.var_list = self.Ws + self.bs
        self.out = self.inp[-1]
        [tf.add_to_collection(tf.GraphKeys.WEIGHTS, _v) for _v in self.Ws]
        [tf.add_to_collection(tf.GraphKeys.BIASES, _v) for _v in self.bs]
        [tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, _v) for _v in self.var_list]
        [tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _v) for _v in self.inp]

    def for_input(self, new_input, new_name=None):
        """
        Returns the same model computed on an other input...

        :param new_input:
        :param new_name:
        :return:
        """
        raise NotImplementedError()

    def _for_input_new_activ_kwargs(self):
        new_active_gen_kwargs = []
        for ag_kw, _W, _b in zip(self.active_gen_kwargs, self.Ws, self.bs):
            n_ag_kw = dict(ag_kw)
            n_ag_kw['init_w'] = _W
            n_ag_kw['init_b'] = _b
            new_active_gen_kwargs.append(n_ag_kw)
        return new_active_gen_kwargs

    def initialize(self, session=None):
        """
        Initialize the model. If `deterministic_initialization` is set to true, 
        saves the initial weight in numpy which will be used for subsequent initialization.
        This is because random seed management in tensorflow is rather obscure... and I could not
        find a way to set the same seed across different initialization without exiting the session.
        
        :param session: 
        :return: 
        """
        ss = session or tf.get_default_session()
        assert ss, 'No default session'
        if not self._var_initializer_op:
            self._var_initializer_op = tf.variables_initializer(self.var_list)
        ss.run(self._var_initializer_op)
        if self.deterministic_initialization:
            if self._var_init_placeholder is None:
                self._var_init_placeholder = tf.placeholder(tf.float32)
                self._assign_int = [v.assign(self._var_init_placeholder) for v in self.var_list]

            if not self._var_list_initial_values:
                self._var_list_initial_values = ss.run(self.var_list)

            else:
                [ss.run(v_op, feed_dict={self._var_init_placeholder: val})
                 for v_op, val in zip(self._assign_int, self._var_list_initial_values)]

    def vectorize(self, *outs, augment=0):
        """
        Calls `vectorize_model` with the variables of this model and specified outputs.
        Moreover it registers this model on the resulting `MergedVariable` and the resulting merged variable 
        in the model as the attribute `self.w`.
        (See `vectorize_model` and `mergedVariable`)

        :param outs: tensors 
        :param augment: 
        :return: 
        """
        res = vectorize_model(self.var_list, *outs, augment=augment)
        res[0].model = self
        self.w = res[0]
        return res


class LinearModel(Network):

    def __init__(self, _input, dim_input, dim_output, name='Linear_Model', deterministic_initialization=False,
                 active_gen=ffnn_layer, **activ_gen_kwargs):
        """
        Builds a single layer NN, by default with linear activation (this means it's just a linear model!)

        :param _input: see `Network`
        :param dim_input: input dimension
        :param dim_output: output dimension
        :param active_gen: callable that genera
        """
        # TODO infer input dimensions form _input....
        super(LinearModel, self).__init__(_input, name,
                                          deterministic_initialization=deterministic_initialization)

        self.dims = (dim_input, dim_output)

        activ_gen_kwargs.setdefault('activ', tf.identity)  # linear model by default
        activ_gen_kwargs.setdefault('init_w', tf.zeros)

        with tf.name_scope(name):
            self.active_gen.append(active_gen)
            self.active_gen_kwargs.append(activ_gen_kwargs)

            ac_func = active_gen(**activ_gen_kwargs)

            _W, _b, _activ = ac_func(self.inp[-1], self.dims)
            self.Ws.append(_W)
            self.bs.append(_b)  # put in the lists
            if dim_output == 1:
                self.inp.append(_activ[:, 0])
            else:
                self.inp.append(_activ)

        self._std_collections()

    def for_input(self, new_input, new_name=None):
        new_active_gen_kwargs = self._for_input_new_activ_kwargs()

        return LinearModel(new_input, self.dims[0], self.dims[1], name=new_name or self.name,
                           active_gen=self.active_gen[0], **new_active_gen_kwargs[0])


class FFNN(Network):
    def __init__(self, _input, dims, name='FFNN', deterministic_initialization=False,
                 active_gen=ffnn_layer, active_gen_kwargs=None
                 ):
        """
        Creates a feed-forward neural network.

        :param _input:
        :param dims:
        :param active_gen:
        :param name:
        """
        super(FFNN, self).__init__(_input, name,
                                   deterministic_initialization=deterministic_initialization)

        pvars(vars())
        self.dims = dims

        active_gen = utils.as_tuple_or_list(active_gen)
        if len(active_gen) != len(dims) - 1:  # assume (hidden, output)
            active_gen = [active_gen[0]] * (len(dims) - 2) + [active_gen[-1]]

        active_gen_kwargs = utils.as_tuple_or_list(active_gen_kwargs or {})
        if len(active_gen_kwargs) != len(dims) - 1:  # assume (hidden, output)
            active_gen_kwargs = [dict(active_gen_kwargs[0]) if active_gen_kwargs[0] else {}] * (len(dims) - 2) \
                                + [dict(active_gen_kwargs[-1]) if active_gen_kwargs[-1] else {}]
        active_gen_kwargs[-1].setdefault('activ', tf.identity)  # sets linear output by default
        active_gen_kwargs[-1].setdefault('init_w', tf.zeros)   # sets weight matrix of last layer to zero by default

        with tf.name_scope(name):
            for d0, d1, ag, ag_kw, l_num in zip(dims, dims[1:], active_gen, active_gen_kwargs, range(len(dims))):
                with tf.name_scope('layer_' + str(l_num)):
                    self.active_gen.append(ag)
                    self.active_gen_kwargs.append(ag_kw)

                    _W, _b, _activ = ag(**ag_kw)(self.inp[-1], [d0, d1])

                    self.Ws.append(_W)
                    self.bs.append(_b)  # put in the lists
                    self.inp.append(_activ)

        self._std_collections()

    # noinspection PyTypeChecker
    def for_input(self, new_input, new_name=None):
        new_active_gen_kwargs = self._for_input_new_activ_kwargs()
        return FFNN(new_input, self.dims, name=new_name or self.name, active_gen=self.active_gen,
                    active_gen_kwargs=new_active_gen_kwargs)


class SimpleConvolutionalOnly(Network):
    def __init__(self, _input, _dims, conv_gen=convolutional_layer2d_maxpool, deterministic_initialization=False,
                 conv_gen_kwargs=None, name='Simple_Convolutional'):
        """
        Creates a simple convolutional network, by default 2 dimensional. Only convolutional part! Use
        `SimpleCNN` for an usual CNN classifier.

        :param _input:
        :param _dims: in default 2d setting, should be a list of quadruples where each quadruple is given by
                        (width, height, # of input channels, # of output channels)
        :param conv_gen:
        :param name:
        """
        super(SimpleConvolutionalOnly, self).__init__(_input, name,
                                                      deterministic_initialization=deterministic_initialization)
        pvars(vars())

        self.dims = _dims

        if not isinstance(conv_gen, (list, tuple)):  # assume all identical
            conv_gen = [conv_gen] * len(_dims)

        if not isinstance(conv_gen_kwargs, (list, tuple)):  # assume all keyword arguments identical
            conv_gen_kwargs = [conv_gen_kwargs or {}] * len(_dims)

        with tf.name_scope(name):

            for sh, ag, ag_kw, l_num in zip(_dims, conv_gen, conv_gen_kwargs, range(len(_dims))):
                with tf.name_scope('layer_' + str(l_num)):
                    self.active_gen.append(ag)
                    self.active_gen_kwargs.append(ag_kw)

                    _W, _b, _out = ag(**ag_kw)(self.inp[-1], sh)

                    self.Ws.append(_W)
                    self.bs.append(_b)  # put in the lists
                    self.inp.append(_out)

        self._std_collections()

    # noinspection PyTypeChecker
    def for_input(self, new_input, new_name=None):
        return SimpleConvolutionalOnly(new_input, _dims=self.dims, conv_gen=self.active_gen,
                                       conv_gen_kwargs=self._for_input_new_activ_kwargs(),
                                       name=new_name or self.name)


class SimpleCNN(Network):

    def __init__(self, _input, conv_part=None, ffnn_part=None, conv_dims=None, ffnn_dims=None,
                 conv_gen=convolutional_layer2d_maxpool, conv_gen_kwargs=None,
                 activ_gen=ffnn_layer, active_gen_kwargs=None, deterministic_initialization=False,
                 name='SimpleCNN'):
        """
        Builds a simple convolutional network a la LeNet5.

        :param _input: input tensor or placeholder (must be given in the right (2d) shape)
        :param conv_part:
        :param ffnn_part:
        :param conv_dims:
        :param ffnn_dims: dimensions for the feed-forward part. Note, the input dimension is inferred
        :param conv_gen:
        :param conv_gen_kwargs:
        :param activ_gen:
        :param active_gen_kwargs:
        :param name:
        """
        assert conv_part or conv_dims
        assert ffnn_part or ffnn_dims

        super(SimpleCNN, self).__init__(_input, name,
                                        deterministic_initialization=deterministic_initialization)
        pvars(vars())

        with tf.name_scope(name):
            self.conv_part = conv_part or SimpleConvolutionalOnly(_input, conv_dims,
                                                                  conv_gen=conv_gen, conv_gen_kwargs=conv_gen_kwargs,
                                                                  name='conv_part')
            self.Ws += self.conv_part.Ws
            self.bs += self.conv_part.bs
            self.inp += self.conv_part.inp
            self.active_gen += self.conv_part.active_gen
            self.active_gen_kwargs += self.conv_part.active_gen_kwargs

            if ffnn_dims:
                ffnn_input = tf.reshape(self.inp[-1], [-1, ffnn_dims[0]])
                ffnn_dims = [ffnn_input.get_shape().as_list()[1]] + ffnn_dims
                self.ffnn_part = FFNN(ffnn_input,
                                      ffnn_dims, active_gen=activ_gen,
                                      active_gen_kwargs=active_gen_kwargs, name='ffnn_part')
            else:
                self.ffnn_part = ffnn_part

            self.Ws += self.ffnn_part.Ws
            self.bs += self.ffnn_part.bs
            self.inp += self.ffnn_part.inp
            self.active_gen += self.ffnn_part.active_gen
            self.active_gen_kwargs += self.ffnn_part.active_gen_kwargs

        self.dims = self.conv_part.dims + self.ffnn_part.dims
        self._std_collections()

    def for_input(self, new_input, new_name=None):
        new_conv = self.conv_part.for_input(new_input)
        new_ffnn_input = tf.reshape(self.inp[-1], [-1, self.ffnn_part.dims[0]])
        new_ffnn = self.ffnn_part.for_input(new_ffnn_input)
        return SimpleCNN(new_input, conv_part=new_conv, ffnn_part=new_ffnn,
                         name=new_name or self.name)

# class SimpleDeCNN(Network):
#     def __init__(self, _input, ffnn_dims, conv_dims,
#                  activ_gen=(ffnn_layer(), ffnn_layer()),
#                  conv_gen=(relu_conv_layer2x2_up_pool, conv_layer),
#                  name='SimpDeCNN'):
#         """_x must be given in the right (2d) shape
#         NOTE: first component of conv_dims must be that of the 2d tensor given as input to the
#         convolutional part of the network"""
#         super(SimpleDeCNN, self).__init__(_input)
#         pvars(vars())
#
#         self.dims = ffnn_dims + conv_dims
#
#         with tf.name_scope(name):
#             ffnn_part = FFNN(_input, ffnn_dims, activ_gen,
#                              name='ffnn_part')
#             self.Ws += ffnn_part.Ws
#             self.bs += ffnn_part.bs
#             self.inp += ffnn_part.inp
#
#             _conv_input = tf.reshape(self.inp[-1], [-1] + conv_dims[0], name='conv_input')
#             if len(conv_gen) != len(conv_dims) - 1:  # assume len(conv_gen) == 2
#                 conv_gen = [conv_gen[0]]*(len(conv_dims) - 2) + [conv_gen[1]]
#             conv_part = SimpleConvolutionalOnly(_conv_input, conv_dims[1:], conv_gen, name='conv_part')
#             self.Ws += conv_part.Ws
#             self.bs += conv_part.bs
#             self.inp += conv_part.inp
#
#         self._std_collections()

if __name__ == '__main__':
    pass
