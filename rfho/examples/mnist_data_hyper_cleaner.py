#!/usr/bin/env python
"""
Script for reproducing the data hyper-cleaning experiment with MNIST data
"""
import argparse
import numpy as np
import tensorflow as tf
import cvxopt
from cvxopt import spmatrix, matrix, sparse

import rfho as ho
from rfho.datasets import load_mnist

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ho.utils_settings['NOTEBOOK_TITLE'] = 'lun_30_50_50_RL0'


class Data:  # what's the purpose of this class???
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def prepare_data(args):
    def limit_to_N_all(_x, _y, _d, _k):
        return _k < args.N_all_

    def count_digit(digit, dset):
        return sum([1 for e in dset if np.argmax(e) == digit])

    mnist = load_mnist(partitions=[args.p_train, args.p_valid], filters=limit_to_N_all)
    data = Data(train=mnist.train, validation=mnist.validation, test=mnist.test)
    if args.verbose:
        for dss in [data.train.target, data.validation.target, data.test.target]:
            for dg in range(10):
                print(dg, count_digit(dg, dss))
            print()
    N_ex = len(data.train.data)
    data.flipped = np.sort(np.random.permutation(list(range(N_ex)))[:args.num_flip])
    # could take also firt args.num_flip examples....

    data.unflipped = list(set(range(N_ex)) - set(data.flipped))
    if args.verbose:
        print('flipped:', data.flipped)
        print('flipped length:', len(data.flipped))
        print(data.train.target[data.flipped][:10])

    for fl in data.flipped:
        new_lab = np.random.choice(list({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} - {np.argmax(data.train.target[fl])}))
        data.train.target[fl] = np.zeros(10)
        data.train.target[fl][new_lab] = 1.

    if args.verbose:
        print(data.train.target[data.flipped][:10])
    return data

# BASELINE EXECUTION
def baseline(args,model,data,x,y,suppliers,lrate):
    error2 = tf.reduce_mean(ho.cross_entropy_loss(model.inp[-1], y))
    correct_prediction2 = tf.equal(tf.argmax(model.inp[-1], 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
    opt = tf.train.GradientDescentOptimizer(lrate)
    ts1 = opt.minimize(error2, var_list=model.var_list)
    with tf.Session(config=config):
        tf.variables_initializer(model.var_list).run()
        for _ in range(args.T):
            ts1.run(feed_dict=suppliers['baseline']())
        baseline_test_accuracy = accuracy2.eval(feed_dict=suppliers['test']())
        return baseline_test_accuracy

# Oracle Execution
def oracle(args,model,data,x,y,suppliers):
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

def hyper_data_cleaner(args,model,data,x,y,suppliers):
    res = ho.vectorize_model(model.var_list, model.inp[-1])
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

    # DOH SETTING
    val_err_dict = {error: [gamma]}
    dynamics = ho.GradientDescentOptimizer.create(w, lr=args.lr, loss=weighted_error)
    doh = ho.ReverseHyperGradient(dynamics, val_err_dict)

    psu = ho.PrintUtils(ho.stepwise_pu(
        lambda ses, step: print('test accuracy',
                                ses.run(accuracy, feed_dict=suppliers['test']())),
        args.T-1),
        ho.stepwise_pu(
            lambda ses, step: print('validation accuracy',
                                    ses.run(accuracy,
                                            feed_dict=suppliers['validation']())),
            args.T-1))
    psu2 = ho.PrintUtils(ho.stepwise_pu(
        lambda ses, step: print('norm of costate',
                                ses.run(ho.norm_p)),
        args.T-1))

    history_test_accuracy = []
    history_validation_accuracy = []
    history_gamma = []

    def save_accuracies(ses, step):
        history_test_accuracy.append(ses.run(accuracy, feed_dict=suppliers['test']()))
        history_validation_accuracy.append(ses.run(accuracy,
                                                   feed_dict=suppliers['validation']()))
        history_gamma.append(ses.run(gamma))


    after_forward_su = ho.PrintUtils(ho.unconditional_pu(save_accuracies))
    grad_hyper = tf.placeholder(tf.float32)

    collected_hyper_gradients = {}  # DO NOT DELETE!
    opt_hyper_dicts = {gamma: ho.AdamOptimizer.create(gamma,
                                               lr=args.hyper_learning_rate,
                                               grad=grad_hyper,
                                               w_is_state=False)}
    gamma_assign = gamma.assign(grad_hyper)

    # Projection
    dim = N_ex
    P = spmatrix(1, range(dim), range(dim))
    glast = matrix(np.ones((1, dim)))
    G = sparse([-P, P, glast])
    h1 = np.zeros(dim)
    h2 = np.ones(dim)
    h = matrix(np.concatenate([h1, h2, [N_unflipped]]))

    def project(gamma):
        print('start projection')
        pt = gamma.eval()
        q = matrix(- np.array(pt, dtype=np.float64))
        # if np.linalg.norm(pt, ord=1) < R:
        #    return
        _res = cvxopt.solvers.qp(P, q, G, h, initvals=q)
        _resx = np.array(_res['x'], dtype=np.float32)[:, 0]
        gamma_assign.eval(feed_dict={grad_hyper: _resx})


    with tf.Session(config=config).as_default() as ss:
        tf.variables_initializer([gamma]).run()

        [ohd.support_variables_initializer().run() for ohd in opt_hyper_dicts.values()]

        for _ in range(args.hyper_iterations):
            if args.verbose:
                print('hyperiteration', _ + 1)
                print('norm 1 of gamma', np.linalg.norm(gamma.eval(), ord=1))
            res = doh.run_all(args.T, train_feed_dict_supplier=suppliers['train'],
                              after_forward_su=after_forward_su,
                              val_feed_dict_suppliers=suppliers['validation'],
                              forward_su=psu, backward_su=None)
            if args.verbose:
                print('end, updating hyperparameters')
            collected_hyper_gradients = ho.ReverseHyperGradient.std_collect_hyper_gradients(res)

            if args.verbose:
                print('hyper gradients')
            ho.print_hyper_gradients(collected_hyper_gradients)

            for hyp in doh.hyper_list:
                ss.run(opt_hyper_dicts[hyp].assign_ops,
                       feed_dict={grad_hyper: collected_hyper_gradients[hyp]})

            project(gamma)
            final_gamma = ss.run(gamma)

            # saving
            last_round_test_accuracy = None
            save_histories = {'history_gamma': history_gamma,
                              'history_validation_accuracy': history_validation_accuracy,
                              'history_test_accuracy': history_test_accuracy,
                              # why saving the same values at each hyper-iteration?
                              # 'baseline_test_accuracy': baseline_test_accuracy,
                              # 'oracle_test_accuracy': oracle_test_accuracy,
                              'last_round_test_accuracy': last_round_test_accuracy,
                              'flipped': data.flipped
                              }
            ho.save_obj(save_histories, 'save_histories', default_overwrite=True)


def main(args):
    data = prepare_data(args)
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

    suppliers = {}
    suppliers['train'] = training_supplier
    suppliers['validation'] = validation_supplier
    suppliers['test'] = test_supplier
    suppliers['baseline'] = baseline_supplier
    suppliers['oracle'] = oracle_supplier

    N_unflipped = int(args.N_all_ * args.p_train - args.num_flip)
    baseline_lrate = args.lr*args.p_train*args.N_all_/N_unflipped
    base_acc = baseline(args,model,data,x,y,suppliers,baseline_lrate)
    oracle_acc = oracle(args,model,data,x,y,suppliers)
    hdc_acc = hyper_data_cleaner(args,model,data,x,y,suppliers)
    print("baseline accuracy: {}".format(base_acc))
    print("oracle accuracy: {}".format(oracle_acc))
    print("hyper-data-cleaner accuracy: {}".format(hdc_acc))


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-H', '--hyper_iterations', type=int, default=2000,
                        help='Number of hyper-iterations')
    parser.add_argument('--T', type=int, default=128,
                        help='Number of parameter optimization iterations')
    parser.add_argument('--N_all_', type=int, default=20000,
                        help='Total number of data points to use')
    parser.add_argument('--num_flip', type=int, default=2500,
                        help='Number of examles with flipped labels')
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
    args = parser.parse_args()
    main(args)
