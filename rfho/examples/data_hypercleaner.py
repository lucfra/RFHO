#!/usr/bin/env python
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

    def _limit_examples(_x, _y, _d, _k):
        return _k < limit_ex

    mnist = load_mnist(folder, partitions=sets_proportions, filters=_limit_examples)

    # ADD NOISE TO LABELS
    noisy_indices = np.sort(np.random.permutation(mnist.train.num_examples)[
                            :int(mnist.train.num_examples*noise)])
    clean_indices = list(set(range(mnist.train.num_examples)) - set(noisy_indices))
    mnist.train.general_info_dict['noisy'] = noisy_indices  # save indices in the Dataset obj
    mnist.train.general_info_dict['clean'] = clean_indices  # save indices in the Dataset obj
    for fl in noisy_indices:  # randomly change targets
        new_lab = np.random.choice(list(set(range(10))
                                        - {np.argmax(mnist.train.target[fl])}))
        mnist.train.target[fl] = np.zeros(10)
        mnist.train.target[fl][new_lab] = 1.

    return mnist


# BASELINE EXECUTION (valid also for oracle)
def baseline(model, y, suppliers, epochs, lr):
    error2 = tf.reduce_mean(ho.cross_entropy_loss(model.inp[-1], y))
    correct_prediction2 = tf.equal(tf.argmax(model.inp[-1], 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
    opt = tf.train.GradientDescentOptimizer(lr)
    ts1 = opt.minimize(error2, var_list=model.var_list)
    with tf.Session(config=rf.CONFIG_GPU_GROWTH):
        tf.variables_initializer(model.var_list).run()
        for _ in range(args.T):
            ts1.run(feed_dict=suppliers['baseline']())
        baseline_test_accuracy = accuracy2.eval(feed_dict=suppliers['test']())
        return baseline_test_accuracy


# Oracle Execution
def oracle(args, model, data, x, y, suppliers):
    error2 = tf.reduce_mean(ho.cross_entropy_loss(model.inp[-1], y))
    correct_prediction2 = tf.equal(tf.argmax(model.inp[-1], 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
    ts2 = tf.train.GradientDescentOptimizer(args.lr).minimize(error2, var_list=model.var_list)
    with tf.Session(config=config):
        tf.variables_initializer(model.var_list).run()
        for _ in range(args.T):
            ts2.run(feed_dict=suppliers['oracle']())
        oracle_test_accuracy = accuracy2.eval(feed_dict=suppliers['test']())
        return oracle_test_accuracy


# HyperLearning execution

def hyper_data_cleaner(args, model, data, x, y, suppliers, l2_regularizer=-2., use_adam=False, R=None):
    res = ho.vectorize_model(model.var_list, model.inp[-1], augment=2*use_adam)
    w = res[0]
    model_out = res[1]

    error = tf.reduce_mean(ho.cross_entropy_loss(model_out, y))

    correct_prediction = tf.equal(tf.argmax(model_out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    N_unflipped = int(args.N_all_ * args.p_train - args.num_flip)
    N_ex = len(data.train.data)
    gamma = tf.Variable(N_unflipped * tf.ones([N_ex]) / N_ex, dtype=tf.float32, name='gamma')

    weighted_error = tf.reduce_mean(gamma * ho.cross_entropy_loss(model_out, y),
                                    name='train_weighted_error')

    alpha = tf.Variable(l2_regularizer, name='alpha')
    regularzation = .5 * tf.exp(alpha) * tf.reduce_sum(tf.pow(w.tensor, 2))

    training_error = weighted_error + regularzation

    # HYPER-OPTIMIZATION SETTING
    val_err_dict = {error: [gamma, alpha]}
    if use_adam:
        dynamics = ho.AdamOptimizer.create(w, loss=training_error)
    else:
        dynamics = ho.GradientDescentOptimizer.create(w, lr=args.lr, loss=training_error)
    hyper_opt = ho.HyperOptimizer(dynamics, val_err_dict, lr=args.hyper_learning_rate)

    def calc_pos_neg(_exw):
        thr = 1.e-3

        exx = np.array(_exw)
        exx[exx < thr] = 0.

        non_zeroed_lambda = set(np.nonzero(exx)[0])
        zeroed_lambda = set(range(len(_exw))) - non_zeroed_lambda
        tp = len(set(data.flipped).intersection(zeroed_lambda))
        fp = len(zeroed_lambda) - tp
        fn = len(set(data.flipped).intersection(non_zeroed_lambda))
        tn = len(non_zeroed_lambda) - fn
        print(tp, fp, fn, tn)
        return tp, fp, fn, tn

    def precision(_exw):
        tp, fp, fn, tn = calc_pos_neg(_exw)
        return tp / (tp + fp) if tp + fp > 0 else 0.

    def recall(_exw):
        tp, fp, fn, tn = calc_pos_neg(_exw)
        return tp / (tp + fn)

    def f1(_exw):
        prec, rec = precision(_exw), recall(_exw)
        return (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0

    grad_hyper = tf.placeholder(tf.float32)
    gamma_assign = gamma.assign(grad_hyper)

    _project = get_projector(N_unflipped, N_ex, R=R)

    def project():
        pt = gamma.eval()
        _resx = _project(pt)
        gamma_assign.eval(feed_dict={grad_hyper: _resx})

    saver = ho.Saver(args.name_of_experiment,
                     'weights', w.tensor,
                     'alpha', alpha,
                     'example weights', gamma,
                     'accuracy train', accuracy, suppliers['train'],
                     'accuracy validation', accuracy, suppliers['validation'],
                     'accuracy test', accuracy, suppliers['test'],
                     'f1', lambda step: f1(gamma.eval()),
                     'precision', lambda step: precision(gamma.eval()),
                     'recall', lambda step: recall(gamma.eval())
                     )

    with tf.Session(config=config).as_default() as ss:

        hyper_opt.initialize()

        for _ in range(args.hyper_iterations):
            if args.verbose:
                print('hyperiteration', _ + 1)
                print('norm 1 of gamma', np.linalg.norm(gamma.eval(), ord=1))
            hyper_opt.initialize()

            hyper_opt.run(args.T, train_feed_dict_supplier=suppliers['train'],
                                val_feed_dict_suppliers=suppliers['validation'])

            project()

            if args.verbose:
                print('end, updating hyperparameters')

            saver.save(_)


def main(args, do_baselines=True, use_adam=False, seed=None, R=None, l2_reg=-4.):
    data = prepare_data(args, seed=seed)
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = ho.LinearModel(x, 28 * 28, 10)

    def training_supplier(step=None):
        return {x: data.train.data, y: data.train.target}

    def validation_supplier(step=None):
        return {x: data.validation.data, y: data.validation.target}

    def test_supplier(step=None):
        return {x: data.test.data, y: data.test.target}

    def baseline_supplier(step=None):
        return {x: np.vstack([data.train.data, data.validation.data]),
                y: np.vstack([data.train.target, data.validation.target])}

    def oracle_supplier(step=None):
        return {x: np.vstack([data.train.data[data.unflipped], data.validation.data]),
                y: np.vstack([data.train.target[data.unflipped], data.validation.target])}

    suppliers = {'train': training_supplier, 'validation': validation_supplier, 'test': test_supplier,
                 'baseline': baseline_supplier, 'oracle': oracle_supplier}

    N_unflipped = int(args.N_all_ * args.p_train - args.num_flip)
    if do_baselines:
        baseline_lrate = args.lr * args.p_train * args.N_all_ / N_unflipped
        base_acc = baseline(args, model, data, x, y, suppliers, baseline_lrate)
        oracle_acc = oracle(args, model, data, x, y, suppliers)
    hdc_acc = hyper_data_cleaner(args, model, data, x, y, suppliers, use_adam=use_adam, R=R, l2_regularizer=l2_reg)
    if do_baselines:
        print("baseline accuracy: {}".format(base_acc))
        print("oracle accuracy: {}".format(oracle_acc))
    # print("hyper-data-cleaner accuracy: {}".format(hdc_acc))


def get_projector(N_unflipped, N_ex, R=None):
    # Projection
    if R is None: R = N_unflipped  # Radius of L1 ball, ..... (should be validated)
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

