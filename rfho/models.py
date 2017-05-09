# import data
# import numpy as np
# working with placeholders
from functools import reduce

import tensorflow as tf
from rfho.utils import MergedVariable
import tensorflow.contrib.graph_editor as ge

test = False
do_print = False

# TODO lot of work to do in this module...
# TODO FFNN part is more mor less fine... instead CNN part is completely to rewrite


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

# this comes form tensorflow tutorials...

def new_weight(_shape, std_dev=.1): return tf.Variable(tf.truncated_normal(_shape, stddev=std_dev))


def new_bias(_shape, _init=.1): return tf.Variable(tf.constant(_init, shape=_shape))  # striding: andare a lunghi passi


def conv2d(_x, _W): return tf.nn.conv2d(_x, _W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(_x): return tf.nn.max_pool(_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def unpool(value, name='unpool'):
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


def conv_layer(_input, _shape, _activ=tf.nn.relu, _stdder=.1, _bias_init=.1):
    pvars(vars(), 1)
    _wc = new_weight(_shape, _stdder)  # [width, height, channes, features ->
    #                                     channels for upper layer] for the sliding window
    _bc = new_bias([_shape[-1]], _bias_init)
    _lin_conv = conv2d(_input, _wc) + _bc
    _h_conv = _activ(_lin_conv, name='conv_activ')
    return _wc, _bc, _h_conv


def relu_conv_layer2x2_max_pool(_input, _shape):
    _wc, _bc, _h_conv = conv_layer(_input, _shape)
    return _wc, _bc, max_pool_2x2(_h_conv)


def relu_conv_layer2x2_up_pool(_input, _shape):
    _wc, _bc, _h_conv = conv_layer(_input, _shape, _stdder=.01, _bias_init=0.)
    return _wc, _bc, unpool(_h_conv)


def relu_conv_layer_2x2_up_pool_dropout(keep_prob):
    def _int(_input, _shape):
        _wc, _bc, _h_conv = conv_layer(_input, _shape, _stdder=.01, _bias_init=0.)
        return _wc, _bc, tf.nn.dropout(unpool(_h_conv), keep_prob)
    return _int


def sigm_conv_layer2x2_up_pool(_input, _shape):
    _wc, _bc, _h_conv = conv_layer(_input, _shape, _activ=tf.nn.sigmoid, _stdder=.1, _bias_init=0.1)
    return _wc, _bc, unpool(_h_conv)


def sig_cnv_layer(_input, _shape):
    _wc, _bc, _h_conv = conv_layer(_input, _shape, _activ=tf.nn.sigmoid, _stdder=.01, _bias_init=0.)
    return _wc, _bc, _h_conv


def tanh_conv_layer(_input, _shape):
    _wc, _bc, _h_conv = conv_layer(_input, _shape, _activ=tf.tanh, _stdder=.01, _bias_init=0.)
    return _wc, _bc, _h_conv


def lin_conv_layer_2x2_up_pool(_input, _shape):
    _wc, _bc, _h_conv = conv_layer(_input, _shape, _activ=tf.identity)
    return _wc, _bc, unpool(_h_conv)


def dropout_activation(_keep_prob, _activ=tf.nn.relu):
    def _int(_v, name='default_name'):
        return tf.nn.dropout(_activ(_v, name), _keep_prob)

    return _int


def create_or_reuse(init_or_variable, shape, name='var'):  # TODO check usage if this function
    # (should be present also in cnn helpers..
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
            proportions = [1/len(activations)]*len(activations)

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
               activ=tf.nn.relu):
    def _int(_input, _shape):
        pvars(vars(), 1)
        _W = create_or_reuse(init_w, _shape, name='W')
        _b = create_or_reuse(init_b, [_shape[1]], name='b')
        _lin_activ = tf.matmul(_input, _W) + _b
        _activ = activ(_lin_activ, name='activation')
        return _W, _b, _activ, activ

    return _int


def ffnn_lin_out(init_w=tf.zeros, init_b=tf.zeros):  # OK
    return ffnn_layer(init_w, init_b, tf.identity)

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


    :param model_vars: list of variables of the model or initializers
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

        new_outs = ge.graph_replace(outs, w.generate_swap_dict())  # FIXME deprecation here on GraphKey usage...

    return [w] + new_outs


class Network(object):

    def __init__(self, _input):
        """
        Creates an object that represent a network. Important attributes of a Network object are

        `var_list`: list of tf.Variables that constitute the parameters of the model

        `inp`: list, first element is `_input` and last should be output of the model. Other entries can be
        hidden layers activations.

        :param _input: tf.Tensor, input of this model.
        """
        super(Network, self).__init__()

        self.Ws = []
        self.bs = []
        self.act_fs = []  # activation functions at each layer
        self.inp = [_input]
        self.var_list = []

    def std_collections(self):
        self.var_list = self.Ws + self.bs
        [tf.add_to_collection(tf.GraphKeys.WEIGHTS, _v) for _v in self.Ws]
        [tf.add_to_collection(tf.GraphKeys.BIASES, _v) for _v in self.bs]
        [tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, _v) for _v in self.var_list]
        [tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, _v) for _v in self.inp]


class LinearModel(Network):

    def __init__(self, _input, dim_input, dim_output,
                 active_gen=ffnn_lin_out()):
        """
        Builds a single layer NN, by default with linear activation (this means it's just a linear model!)

        :param _input: see `Network`
        :param dim_input: input dimension
        :param dim_output: output dimension
        :param active_gen: callable that genera
        """
        # TODO infer input and output dimensions form input....
        super(LinearModel, self).__init__(_input)

        self.dims = [dim_input, dim_output]

        with tf.name_scope('model_parameters'):
            _W, _b, _activ, act_f = active_gen(self.inp[-1], [dim_input, dim_output])
            self.Ws.append(_W)
            self.bs.append(_b)  # put in the lists
            self.inp.append(_activ)
            self.act_fs.append(act_f)

        self.std_collections()


class FFNN(Network):

    def __init__(self, _input, dims,
                 activ_gen=(ffnn_layer(), ffnn_lin_out()),
                 name='FFNN'):
        """
        Creates a feed-forward neural network.

        :param _input:
        :param dims:
        :param activ_gen:
        :param name:
        """
        # TODO infer input and output dimensions form input....
        super(FFNN, self).__init__(_input)

        pvars(vars())
        self.dims = dims

        if len(activ_gen) != len(dims) - 1:  # assume (hidden, output)
            activ_gen = [activ_gen[0]] * (len(dims) - 2) + [activ_gen[1]]

        with tf.name_scope(name):
            for d0, d1, ag, l_num in zip(dims, dims[1:], activ_gen, range(len(dims))):
                with tf.name_scope('layer_' + str(l_num)):
                    _W, _b, _activ, act_f = ag(self.inp[-1], [d0, d1])

                    self.Ws.append(_W)
                    self.bs.append(_b)  # put in the lists
                    self.inp.append(_activ)
                    self.act_fs.append(act_f)  # store also activation functions (might be useful..)

        self.std_collections()

        # FFNN class end ###


class SimpleConvolutionalOnly(Network):  # TODO STORE ACTIVATION FUNCTIONS

    def __init__(self, _input, _dims, conv_gen=relu_conv_layer2x2_max_pool,
                 name='Simple_Convolutional'):
        """
        Creates a simple convolutional network, by default 2 dimensional. Only convolutional part! Use
        `SimpleCNN` for an usual CNN classifier.

        :param _input:
        :param _dims: in default 2d setting, should be a list of quadruples where each quadruple is given by
                        (width, height, # of input channels, # of output channels)
        :param conv_gen:
        :param name:
        """
        super(SimpleConvolutionalOnly, self).__init__(_input)
        pvars(vars())

        self.dims = _dims

        with tf.name_scope(name):
            if not isinstance(conv_gen, (list, tuple)):  # assume all identical
                conv_gen = [conv_gen] * len(_dims)

            for sh, ag, l_num in zip(_dims, conv_gen, range(len(_dims))):
                with tf.name_scope('layer_' + str(l_num)):
                    _W, _b, _out = ag(self.inp[-1], sh)

                    self.Ws.append(_W)
                    self.bs.append(_b)  # put in the lists
                    self.inp.append(_out)

        self.std_collections()


class SimpleCNN(Network):  # TODO check that class works fine... # STORE ACTIVATION FUNCTIONS

    def __init__(self, _input, conv_dims, ffnn_dims,
                 conv_gen=relu_conv_layer2x2_max_pool,
                 activ_gen=(ffnn_layer(), ffnn_lin_out()),
                 name='SimpCNN'):
        """_input must be given in the right (2d) shape"""
        super(SimpleCNN, self).__init__(_input)
        pvars(vars())

        self.dims = conv_dims + ffnn_dims

        with tf.name_scope(name):
            conv_part = SimpleConvolutionalOnly(_input, conv_dims, conv_gen, name='conv_part')
            self.Ws += conv_part.Ws
            self.bs += conv_part.bs
            self.inp += conv_part.inp

            ffnn_dims[0] *= conv_dims[-1][-1]  # adjust ffnn inp. dim to consider the channels in the last conv layer
            ffnn_part = FFNN(tf.reshape(self.inp[-1], [-1, ffnn_dims[0]]),
                             ffnn_dims, activ_gen,
                             name='ffnn_part')
            self.Ws += ffnn_part.Ws
            self.bs += ffnn_part.bs
            self.inp += ffnn_part.inp

        self.std_collections()


class SimpleDeCNN(Network):
    def __init__(self, _input, ffnn_dims, conv_dims,
                 activ_gen=(ffnn_layer(), ffnn_layer()),
                 conv_gen=(relu_conv_layer2x2_up_pool, conv_layer),
                 name='SimpDeCNN'):
        """_x must be given in the right (2d) shape
        NOTE: first component of conv_dims must be that of the 2d tensor given as input to the
        convolutional part of the network"""
        super(SimpleDeCNN, self).__init__(_input)
        pvars(vars())

        self.dims = ffnn_dims + conv_dims

        with tf.name_scope(name):
            ffnn_part = FFNN(_input, ffnn_dims, activ_gen,
                             name='ffnn_part')
            self.Ws += ffnn_part.Ws
            self.bs += ffnn_part.bs
            self.inp += ffnn_part.inp

            _conv_input = tf.reshape(self.inp[-1], [-1] + conv_dims[0], name='conv_input')
            if len(conv_gen) != len(conv_dims) - 1:  # assume len(conv_gen) == 2
                conv_gen = [conv_gen[0]]*(len(conv_dims) - 2) + [conv_gen[1]]
            conv_part = SimpleConvolutionalOnly(_conv_input, conv_dims[1:], conv_gen, name='conv_part')
            self.Ws += conv_part.Ws
            self.bs += conv_part.bs
            self.inp += conv_part.inp

        self.std_collections()
