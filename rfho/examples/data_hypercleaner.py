#!/usr/bin/env python
# FIXME  new version: not yet tested
"""
Script for reproducing the data hyper-cleaning experiment with MNIST data
"""
import argparse
import numpy as np
import tensorflow as tf
import rfho as rf
import cvxopt
from cvxopt import spmatrix, matrix, sparse

from rfho.datasets import load_mnist


def load_std_data(folder=None, limit_ex=20000, sets_proportions=(.25, .25),
                  noise=.5):
    """
    If necessary, set the seed before calling this funciton

    :param folder:
    :param limit_ex:
    :param sets_proportions:
    :param noise: noise level
    :return:
    """

    # noinspection PyUnusedLocal
    def _limit_examples(_x, _y, _d, _k):
        return _k < limit_ex

    mnist = load_mnist(folder, partitions=sets_proportions, filters=_limit_examples)

    # ADD NOISE TO LABELS
    noisy_indices = np.sort(np.random.permutation(mnist.train.num_examples)[
                            :int(mnist.train.num_examples*noise)])
    clean_indices = list(set(range(mnist.train.num_examples)) - set(noisy_indices))

    for fl in noisy_indices:  # randomly change targets
        new_lab = np.random.choice(list(set(range(10))
                                        - {np.argmax(mnist.train.target[fl])}))
        mnist.train.target[fl] = np.zeros(10)
        mnist.train.target[fl][new_lab] = 1.

    info_dict = {
        'noisy examples': noisy_indices,
        'clean examples': clean_indices,
        'N noisy': len(noisy_indices),
        'N clean': len(clean_indices)
    }

    mnist.train.general_info_dict = info_dict  # save some useful stuff

    return mnist


def baseline(saver, model, y, data, T, lr, lmd=None, name='baseline'):
    # TODO other optimizers?
    """
    BASELINE EXECUTION (valid also for oracle and final training,
    with optimized values of lambda)

    :param saver: `Saver` object (can be None)
    :param name: optional name for the saver
    :param data: `Datasets` object
    :param T: number of iterations
    :param lmd: weights for the examples, if None sets to 1.
    :param model: a model (should comply with `rf.Network`)
    :param y: placeholder for output
    :param lr: learning rate
    :return:
    """
    # TODO use also saver...
    x = model.inp[0]

    def train_and_valid_s():
        return {x: np.vstack((data.train.data, data.validation.data)),
                y: np.vstack((data.train.target, data.validation.target))}
    tst_s = data.test.create_supplier(x, y)

    if not lmd: lmd = np.ones(train_and_valid_s()[x].shape)

    error2 = tf.reduce_mean(lmd * rf.cross_entropy_loss(y, model.out))
    correct_prediction2 = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

    opt = tf.train.GradientDescentOptimizer(lr)
    ts1 = opt.minimize(error2, var_list=model.var_list)

    # saver related
    if saver:
        saver.clear_items()
        saver.add_items(
            # TODO
        )

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():
        tf.variables_initializer(model.var_list).run()
        for _ in range(T):
            ts1.run(feed_dict=train_and_valid_s())
        if saver: saver.save(name)
        baseline_test_accuracy = accuracy2.eval(feed_dict=tst_s())
        return baseline_test_accuracy


def data_hyper_cleaner(saver, model, y, data, T, lr, R,
                       optimizer=rf.GradientDescentOptimizer,
                       optimizer_kwargs=None,
                       hyper_iterations=500, hyper_grad_kwargs=None,
                       hyper_optimizer_class=rf.AdamOptimizer,
                       hyper_optimizer_kwargs=None):
    """

    :param saver: `Saver` object (can be None)
    :param data: `Datasets` object
    :param T: number of iterations
    :param model: a model (should comply with `rf.Network`)
    :param y: placeholder for output
    :param lr: learning rate
    :param R: radius of L1 ball
    :param optimizer: parameter optimizer
    :param optimizer_kwargs: optional arguments for parameter optimizer
    :param hyper_iterations: number of hyper-iterations
    :param hyper_grad_kwargs: optional arguments for `ReverseHG` (such as weight history)
    :param hyper_optimizer_class: optimizer class for hyperparameters
    :param hyper_optimizer_kwargs: optional arguments for hyperparameter optimizer
    :return:
    """

    if hyper_optimizer_kwargs is None:
        hyper_optimizer_kwargs = {}

    x = model.inp[0]
    w, out = rf.vectorize_model(model.var_list, model.out,
                                augment=optimizer.get_augmentation_multiplier())

    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), "float"))

    sample_error = rf.cross_entropy_loss(y, out)
    error = tf.reduce_mean(sample_error)

    lmd = tf.Variable(tf.ones([data.train.num_examples]) / R,
                      dtype=tf.float32, name='lambda')

    weighted_error = tf.reduce_mean(tf.multiply(lmd, sample_error),
                                    name='train_weighted_error')
    # ADD REGULARIZATION??
    dynamics = optimizer.create(w, lr=lr, loss=weighted_error, **optimizer_kwargs or {})

    hyper_opt = rf.HyperOptimizer(dynamics, {error: lmd}, method=rf.ReverseHG,
                                  hyper_grad_kwargs=hyper_grad_kwargs or {},
                                  hyper_optimizer_class=hyper_optimizer_class,
                                  **hyper_optimizer_kwargs
                                  )

    # projection
    grad_hyper = tf.placeholder(tf.float32)
    lmd_assign = lmd.assign(grad_hyper)

    _project = get_projector(R=R, N_ex=data.train.num_examples)

    def project():
        pt = lmd.eval()
        _resx = _project(pt)
        lmd_assign.eval(feed_dict={grad_hyper: _resx})

    # suppliers
    tr_s = data.train.create_supplier(x, y)
    val_s = data.validation.create_supplier(x, y)
    tst_s = data.test.create_supplier(x, y)

    if saver:
        saver.clear_items()  # just to be sure!
        saver.add_items(
            'validation accuracy', accuracy, val_s,
            'lambda', lmd,
            # TODO etc... with precision, recall f1....
        )

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():
        for hyt in range(hyper_iterations):
            hyper_opt.initialize()
            hyper_opt.run(T, train_feed_dict_supplier=tr_s,
                          val_feed_dict_suppliers={error: val_s},
                          hyper_constraints_ops=project)
            if saver: saver.save(hyt)

# TODO check the following functions

def get_projector(R, N_ex):  # !
    # Projection
    dim = N_ex
    P = spmatrix(1, range(dim), range(dim))
    glast = matrix(np.ones((1, dim)))
    G = sparse([-P, P, glast])
    h1 = np.zeros(dim)
    h2 = np.ones(dim)
    h = matrix(np.concatenate([h1, h2, [R]]))

    def project(pt):
        print('start projection')
        # pt = gamma.eval()
        q = matrix(- np.array(pt, dtype=np.float64))
        # if np.linalg.norm(pt, ord=1) < R:
        #    return
        _res = cvxopt.solvers.qp(P, q, G, h, initvals=q)
        _resx = np.array(_res['x'], dtype=np.float32)[:, 0]
        # gamma_assign.eval(feed_dict={grad_hyper: _resx})
        return _resx

    return project


def calc_pos_neg(_exw, data):
    thr = 1.e-3

    exx = np.array(_exw)
    exx[exx < thr] = 0.

    non_zeroed_lambda = set(np.nonzero(exx)[0])  # does this work???
    zeroed_lambda = set(range(len(_exw))) - non_zeroed_lambda
    tp = len(set(data.flipped).intersection(zeroed_lambda))
    fp = len(zeroed_lambda) - tp
    fn = len(set(data.flipped).intersection(non_zeroed_lambda))
    tn = len(non_zeroed_lambda) - fn
    print(tp, fp, fn, tn)
    return tp, fp, fn, tn


def precision(_exw, data):
    tp, fp, fn, tn = calc_pos_neg(_exw, data)
    return tp / (tp + fp) if tp + fp > 0 else 0.


def recall(_exw, data):
    tp, fp, fn, tn = calc_pos_neg(_exw, data)
    return tp / (tp + fn)


def f1(_exw):
    prec, rec = precision(_exw), recall(_exw)
    return (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0
# end


def main(args, do_baselines=True, use_adam=False, seed=None, R=None, l2_reg=-4.):
    # TODO ...
    data = load_std_data()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = rf.LinearModel(x, 28 * 28, 10)

    if do_baselines:
        pass
        # baseline_lrate = args.lr * args.p_train * args.N_all_ / N_unflipped
        # base_acc = baseline(args, model, data, x, y, suppliers, baseline_lrate)
        # oracle_acc = oracle(args, model, data, x, y, suppliers)

    # optimize ...
    # hdc_acc = hyper_data_cleaner(args, model, data, x, y, suppliers, use_adam=use_adam, R=R, l2_regularizer=l2_reg)
    # if do_baselines:
    #     print("baseline accuracy: {}".format(base_acc))
    #     print("oracle accuracy: {}".format(oracle_acc))
    # print("hyper-data-cleaner accuracy: {}".format(hdc_acc))



# sure we want to use this parser? it's kind of uncomfortable..
def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--name_of_experiment', type=str, default='hypercleaner',
                        help='Name of the experiment')
    parser.add_argument('-H', '--hyper_iterations', type=int, default=500,
                        help='Number of hyper-iterations')
    parser.add_argument('--T', type=int, default=1000,
                        help='Number of parameter optimization iterations')
    parser.add_argument('--N_all_', type=int, default=20000,
                        help='Total number of data points to use')
    parser.add_argument('--num_flip', type=int, default=2500,
                        help='Number of examples with wrong labels')
    parser.add_argument('--p_train', type=float, default=0.25,
                        help='Fraction of training data points')
    parser.add_argument('--p_valid', type=float, default=0.25,
                        help='Fraction of validation data points')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--hyper_learning_rate', type=float, default=.005,
                        help='Hyper learning rate')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Verbose output.')
    parser.set_defaults(verbose=False)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    # args = parser.parse_args()
    # args.name_of_experiment = 'hypercleaner_normal_R'
    # main(args, do_baselines=False, use_adam=False, seed=0, R=None, l2_reg=-5.)

