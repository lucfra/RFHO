import numpy as np
import tensorflow as tf
import cvxopt
from cvxopt import spmatrix, matrix, sparse

import rfho as ho

from rfho.datasets import load_mnist

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

ho.settings['NOTEBOOK_TITLE'] = 'lun_30_50_50_RL0'

N_all_ = 20000
p_train = .25
p_valid = .25

lr = .1

hyper_iterations = 2000
T = 128
num_flip = 2500

hyper_learning_rate = .005  # default is 0.001

N_unflipped = int(N_all_ * p_train - num_flip)


def limit_to_N_all(_x, _y, _d, _k):
    return _k < N_all_


data = load_mnist(partitions=[p_train, p_valid], filters=limit_to_N_all)

def count_digit(digit, dset):
    return sum([1 for e in dset if np.argmax(e) == digit])


for dss in [data.train.target, data.validation.target, data.test.target]:
    for dg in range(10):
        print(dg, count_digit(dg, dss))
    print()

N_ex = len(data.train.data)

flipped = np.sort(np.random.permutation(list(range(N_ex)))[:num_flip])  # could take also firt num_flip examples....
unflipped = list(set(range(N_ex)) - set(flipped))
print('flipped:', flipped)
print('flipped length:', len(flipped))
print(data.train.target[flipped][:10])

for fl in flipped:
    new_lab = np.random.choice(list({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} - {np.argmax(data.train.target[fl])}))
    data.train.target[fl] = np.zeros(10)
    data.train.target[fl][new_lab] = 1.

print(data.train.target[flipped][:10])

# imgs = np.hstack([np.reshape(e, (28, 28)) for e in data.train.data[flipped][:10]])
# plt.matshow(imgs)
# plt.grid()
# plt.show()
# print([np.argmax(e) for e in data.train.target[flipped][:10]])

# MODEL

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
model = ho.LinearModel(x, 28 * 28, 10)

res = ho.vectorize_model(model.var_list, model.inp[-1])
w = res[0]
model_out = res[1]

error = tf.reduce_mean(ho.cross_entropy_loss(model_out, y))

correct_prediction = tf.equal(tf.argmax(model_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

gamma = tf.Variable(N_unflipped * tf.ones([N_ex]) / N_ex, dtype=tf.float32, name='gamma')

weighted_error = tf.reduce_mean(gamma * ho.cross_entropy_loss(model_out, y), name='train_weighted_error')

# DOH SETTING

val_err_dict = {error: [gamma]}

dynamics = ho.gradient_descent(w, lr=lr, loss=weighted_error)

doh = ho.ReverseHyperGradient(dynamics, val_err_dict)


# ev = ExampleVisiting(data, batch_size, 100)

# training_supplier = ev.training_supplier(x,y)
# validation_supplier = ev.all_validation_supplier(x,y)
# test_supplier = ev.all_test_supplier(x,y)

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
    return {x: np.vstack([data.train.data[unflipped],
                          data.validation.data]),
            y: np.vstack([data.train.target[unflipped],
                          data.validation.target])}


psu = ho.PrintUtils(ho.stepwise_pu(
    lambda ses, step: print('test accuracy', ses.run(accuracy, feed_dict=test_supplier())), T - 1),
    ho.stepwise_pu(
        lambda ses, step: print('validation accuracy', ses.run(accuracy, feed_dict=validation_supplier())), T - 1))
psu2 = ho.PrintUtils(ho.stepwise_pu(
    lambda ses, step: print('norm of costate', ses.run(ho.norm_p)), T - 1))

history_test_accuracy = []
history_validation_accuracy = []
history_gamma = []


def save_accuracies(ses, step):
    history_test_accuracy.append(ses.run(accuracy, feed_dict=test_supplier()))
    history_validation_accuracy.append(ses.run(accuracy, feed_dict=validation_supplier()))
    history_gamma.append(ses.run(gamma))


after_forward_su = ho.PrintUtils(ho.unconditional_pu(save_accuracies))

grad_hyper = tf.placeholder(tf.float32)

collected_hyper_gradients = {}  # DO NOT DELETE!

opt_hyper_dicts = {gamma: ho.adam_dynamics(gamma, lr=hyper_learning_rate, grad=grad_hyper, w_is_state=False)}

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


# BASELINE EXECUTION

error2 = tf.reduce_mean(ho.cross_entropy_loss(model.inp[-1], y))

correct_prediction2 = tf.equal(tf.argmax(model.inp[-1], 1), tf.argmax(y, 1))
accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

ts1 = tf.train.GradientDescentOptimizer(lr * p_train * N_all_ / N_unflipped).minimize(error2, var_list=model.var_list)

with tf.Session(config=config):
    tf.variables_initializer(model.var_list).run()
    for _ in range(T):
        ts1.run(feed_dict=baseline_supplier())
    baseline_test_accuracy = accuracy2.eval(feed_dict=test_supplier())

    print("baseline accuracy: {}".format(baseline_test_accuracy))

# Oracle Execution

ts2 = tf.train.GradientDescentOptimizer(lr).minimize(error2, var_list=model.var_list)

with tf.Session(config=config):
    tf.variables_initializer(model.var_list).run()
    for _ in range(T):
        ts2.run(feed_dict=oracle_supplier())
    oracle_test_accuracy = accuracy2.eval(feed_dict=test_supplier())

    print("oracle test accuracy: {}".format(oracle_test_accuracy))

# saving

last_round_test_accuracy = None


def save_all():
    save_histories = {
        'history_gamma': history_gamma, 'history_validation_accuracy': history_validation_accuracy,
        'history_test_accuracy': history_test_accuracy, 'baseline_test_accuracy': baseline_test_accuracy,
        'oracle_test_accuracy': oracle_test_accuracy, 'last_round_test_accuracy': last_round_test_accuracy,
        'flipped': flipped
    }
    ho.save_obj(save_histories, 'save_histories', default_overwrite=True)


# HyperLearning execution

with tf.Session(config=config).as_default() as ss:
    tf.variables_initializer([gamma]).run()

    [ohd.support_variables_initializer().run() for ohd in opt_hyper_dicts.values()]

    for _ in range(hyper_iterations):

        print('hyperiteration', _ + 1)
        print('norm 1 of gamma', np.linalg.norm(gamma.eval(), ord=1))
        res = doh.run_all(T, training_supplier=training_supplier, after_forward_su=after_forward_su,
                          validation_suppliers=validation_supplier, forward_su=psu, backward_su=None)

        print('end, updating hyperparameters')

        collected_hyper_gradients = ho.ReverseHyperGradient.std_collect_hyper_gradients(res)

        print('hyper gradients')
        ho.print_hyper_gradients(collected_hyper_gradients)

        for hyp in doh.hyper_list:
            ss.run(opt_hyper_dicts[hyp].assign_ops, feed_dict={grad_hyper: collected_hyper_gradients[hyp]})


        project(gamma)
        final_gamma = ss.run(gamma)

        save_all()

        print()
