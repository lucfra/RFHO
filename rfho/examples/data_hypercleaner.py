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
                            :int(mnist.train.num_examples * noise)])
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

    mnist.train.info = info_dict  # save some useful stuff

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
    # if saver: saver.save_setting(vars(), append_string=name)
    x = model.inp[0]

    # def _train_and_valid_s():
    #     return {x: np.vstack((data.train.data, data.validation.data)),
    #             y: np.vstack((data.train.target, data.validation.target))}
    train_and_valid = rf.datasets.Dataset.stack(data.train, data.validation)
    train_and_valid_s = train_and_valid.create_supplier(x, y)

    tst_s = data.test.create_supplier(x, y)

    if lmd is None: lmd = np.ones(train_and_valid.num_examples)

    error2 = tf.reduce_mean(lmd * rf.cross_entropy_loss(y, model.out))
    correct_prediction2 = tf.equal(tf.argmax(model.out, 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

    opt = tf.train.GradientDescentOptimizer(lr)
    ts1 = opt.minimize(error2, var_list=model.var_list)

    # saver related
    if saver:
        saver.clear_items()
        saver.add_items(
            'Test Accuracy', accuracy2, tst_s,
        )

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():
        tf.variables_initializer(model.var_list).run()
        for _ in range(T):
            ts1.run(feed_dict=train_and_valid_s())
        if saver: saver.save(name)
        baseline_test_accuracy = accuracy2.eval(feed_dict=tst_s())
        return baseline_test_accuracy


def data_hyper_cleaner(saver, model, y, data, T, lr, R,
                       toy_problem=True, append_string='',
                       optimizer=rf.GradientDescentOptimizer,
                       optimizer_kwargs=None,
                       hyper_iterations=500, hyper_grad_kwargs=None,
                       hyper_optimizer_class=rf.AdamOptimizer,
                       hyper_optimizer_kwargs=None):
    """

    :param append_string: string to append for file saving.
    :param toy_problem: if True computes _precision _recall and _f1. (in a real problem this would not be feasible..)
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
    if saver: saver.save_setting(vars())

    if hyper_optimizer_kwargs is None:
        hyper_optimizer_kwargs = {}

    x = model.inp[0]
    w, out = rf.vectorize_model(model.var_list, model.out,
                                augment=optimizer.get_augmentation_multiplier())

    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), "float"))

    sample_error = rf.cross_entropy_loss(y, out)
    error = tf.reduce_mean(sample_error)

    lmd = tf.Variable(R * tf.ones([data.train.num_examples]) / data.train.num_examples,
                      dtype=tf.float32, name='lambda')

    weighted_error = tf.reduce_mean(tf.multiply(lmd, sample_error),
                                    name='train_weighted_error')
    # ADD REGULARIZATION??  (shouldn't be necessary)
    dynamics = optimizer.create(w, lr=lr, loss=weighted_error, **optimizer_kwargs or {})

    hyper_opt = rf.HyperOptimizer(dynamics, {error: lmd}, method=rf.ReverseHG,
                                  hyper_grad_kwargs=hyper_grad_kwargs or {},
                                  hyper_optimizer_class=hyper_optimizer_class,
                                  **hyper_optimizer_kwargs
                                  )

    # projection
    grad_hyper = tf.placeholder(tf.float32)
    lmd_assign = lmd.assign(grad_hyper)

    _project = _get_projector(R=R, N_ex=data.train.num_examples)

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
        )
        if toy_problem:
            saver.add_items(
                'tp, fp, fn, tn', lambda stp: _calc_pos_neg(lmd.eval(), data),
                'Precision', lambda stp: _precision(lmd.eval(), data),
                'Recall', lambda stp: _recall(lmd.eval(), data),
                'F1', lambda stp: _f1(lmd.eval(), data),
            )

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():
        for hyt in range(hyper_iterations):
            hyper_opt.initialize()
            hyper_opt.run(T, train_feed_dict_supplier=tr_s,
                          val_feed_dict_suppliers={error: val_s},
                          hyper_constraints_ops=project)
            if saver: saver.save(hyt, append_string=append_string)
        saver.pack_save_dictionaries(append_string=append_string)  # zips all
        return lmd.eval()


def _get_projector(R, N_ex):  # !
    # Projection
    dim = N_ex
    P = spmatrix(1, range(dim), range(dim))
    glast = matrix(np.ones((1, dim)))
    G = sparse([-P, P, glast])
    h1 = np.zeros(dim)
    h2 = np.ones(dim)
    h = matrix(np.concatenate([h1, h2, [R]]))

    def _project(pt):
        print('start projection')
        # pt = gamma.eval()
        q = matrix(- np.array(pt, dtype=np.float64))
        # if np.linalg.norm(pt, ord=1) < R:
        #    return
        _res = cvxopt.solvers.qp(P, q, G, h, initvals=q)
        _resx = np.array(_res['x'], dtype=np.float32)[:, 0]
        # gamma_assign.eval(feed_dict={grad_hyper: _resx})
        return _resx

    return _project


# TODO check the following functions (look right)


def _calc_pos_neg(_exw, data, thr=1.e-3):
    exx = np.array(_exw)
    exx[exx < thr] = 0.

    # noinspection PyUnresolvedReferences
    non_zeroed_lambda = set(np.nonzero(exx)[0])  # does this work??? looks like.. my PyCharm warns me
    zeroed_lambda = set(range(len(_exw))) - non_zeroed_lambda
    tp = len(set(data.train.info['noisy examples']).intersection(zeroed_lambda))
    fp = len(zeroed_lambda) - tp
    fn = len(set(data.train.info['noisy examples']).intersection(non_zeroed_lambda))
    tn = len(non_zeroed_lambda) - fn
    return tp, fp, fn, tn


def _precision(_exw, data):
    tp, fp, fn, tn = _calc_pos_neg(_exw, data)
    return tp / (tp + fp) if tp + fp > 0 else 0.


def _recall(_exw, data):
    tp, fp, fn, tn = _calc_pos_neg(_exw, data)
    return tp / (tp + fn)


def _f1(_exw, data):
    prec, rec = _precision(_exw, data), _recall(_exw, data)
    return (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0


# end


def main(saver=None, run_baseline=True, run_oracle=True, run_optimization=True,
         T=2000, lr=.1, R_search_space=(1000, 1500, 2000, 2500)):
    # TODO ...
    """
    This method should replicate ICML experiment....
    
    :param R_search_space: 
    :param saver: 
    :param run_baseline: 
    :param run_oracle: 
    :param run_optimization: 
    :param T: 
    :param lr: 
    :return: 
    """
    data = load_std_data()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = rf.LinearModel(x, 28 * 28, 10)

    if run_baseline:
        # baseline_lrate = args.lr * args.p_train * args.N_all_ / N_unflipped  # better lr of optimizer i'd say
        baseline(saver, model, y, data, T, lr, name='Baseline')  # could run for more iterations...

    if run_oracle:
        lmd_oracle = np.ones(data.train.num_examples + data.validation.num_examples)
        lmd_oracle[data.train.info['noisy examples']] = 0.
        # for ind in data.train.info['noisy examples']:
        #     lmd_oracle[ind] = 0.
        baseline(saver, model, y, data, T, lr, lmd=lmd_oracle, name='Oracle')

    lmd_dict = {}
    if run_optimization:
        for R in R_search_space:
            normalized_lr = lr * data.train.num_examples / R
            lmd_dict[R] = data_hyper_cleaner(saver, model, y, data, T, lr=normalized_lr, R=R,
                                             hyper_optimizer_kwargs={'lr': 0.005},
                                             # i think we used this one in the paper
                                             append_string='_R%d' % R)
            lmd_optimized = np.ones(data.train.num_examples + data.validation.num_examples)
            lmd_optimized *= np.max(lmd_dict[R])  # there's no reason to overweight the validation examples...
            lmd_optimized[:data.train.num_examples] = lmd_dict[R]
            baseline(saver, model, y, data, T, lr=.1 / np.max(lmd_dict[R]), lmd=lmd_optimized,
                     name='_R%d_Final Round' % R)  # could run for more iterations...


def get_parser():  # sure we want to use this parser? it's kind of uncomfortable..
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


def quick_demo(lmd=None):
    """
    Just to show that it works...
    
    :return: 
    """
    saver = rf.Saver(['Data Hyper-cleaner', 'Quick Demo'])
    np.random.seed(0)

    data = load_std_data()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = rf.LinearModel(x, 28 * 28, 10)

    if lmd is None:
        # the behaviour in this particular setting is quite stable, we can reduce the times by setting small R,
        # higher hyper-learning rates and a small number of training iterations (T) and hyper-iterations
        lmd = data_hyper_cleaner(saver, model, data=data, y=y, T=500, lr=.1, R=1000.,
                                 toy_problem=True,
                                 hyper_iterations=30, hyper_optimizer_kwargs={'lr': 0.01})

    lmd_optimized = np.ones(data.train.num_examples + data.validation.num_examples)
    lmd_optimized *= np.max(lmd)  # there's no reason to overweight the validation examples...
    lmd_optimized[:data.train.num_examples] = lmd
    baseline(saver, model, y, data, 4000, lr=.1 / np.max(lmd), lmd=lmd_optimized, name='Final Round')


if __name__ == '__main__':
    # saved = rf.load_obj('29', root_dir='/media/luca/DATA/EXPERIMENTS/Data Hyper-cleaner/Quick Demo/22-06-17__15h39m')
    # print(saved)  # at first I forgot a thing...
    quick_demo(
        # saved['lambda']
    )

    # parser = get_parser()
    # args = parser.parse_args()
    # args.name_of_experiment = 'hypercleaner_normal_R'
    # main(rf.Saver('TBD'), run_baseline=False, T=4000, run_optimization=False)
