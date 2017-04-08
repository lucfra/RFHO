import tensorflow as tf
import numpy as np
from rfho.models import LinearModel, vectorize_model
from rfho.utils import cross_entropy_loss, stepwise_pu, unconditional_pu, PrintUtils, binary_cross_entropy,\
    simple_name, Vl_Mode
from rfho.save_and_load import save_obj, settings, load_obj
from rfho.hyper_gradients import ReverseHyperGradient, adam_dynamics, print_hyper_gradients
from rfho.optimizers import momentum_dynamics, adam_dynamics

import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def single_task_learning(name_of_experiment, datasets, class_num,
                         T=3000, learning_rate=.01, momentum=.9, collect_data=True,
                         rho_search_space=np.linspace(0., 1.e-1)):

    settings['NOTEBOOK_TITLE'] = name_of_experiment

    train_x, val_x = datasets.train.data, datasets.validation.data

    def mod_labels(dat):
        res = np.zeros([dat.num_examples, 1])
        for i, e in enumerate(dat.target):
            res[i] = (np.argmax(e) == class_num)
        return res

    train_y, val_y = mod_labels(datasets.train), mod_labels(datasets.validation)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = LinearModel(x, datasets.train.dim_data, 1)

    out, W_mod, b_mod = (model.inp[-1], model.Ws[0], model.bs[0])

    correct_prediction = tf.equal((tf.sign(tf.nn.sigmoid(out)[:, 0] - .5) + 1.)/2., y[:, 0])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    rho = tf.placeholder(tf.float32, name='rho')
    error = tf.reduce_mean(binary_cross_entropy(out, y)) + rho*tf.reduce_sum(W_mod**2)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

    ts = optimizer.minimize(error, var_list=model.var_list)

    def tr_s():
        nonlocal val_rho
        return {x: train_x, y: train_y, rho: val_rho}

    def val_s():
        return {x: val_x, y: val_y}

    def val_and_train():
        return {x: np.vstack([train_x, val_x]), y: np.vstack([train_y, val_y]), rho: val_rho}

    def do_some_saves(append_name=''):
        nonlocal val_rho
        if not collect_data:
            return
        name = 'class_%d_rho_%.4e%s' % (class_num, val_rho, append_name)
        save_dict = {
            'validation accuracy': accuracy.eval(feed_dict=val_s()),
            'training accuracy': accuracy.eval(feed_dict=tr_s()),
            'rho': val_rho,
            'W': W_mod.eval(),
            'b': b_mod.eval(),
        }
        save_obj(save_dict, name)
        return save_dict

    # noinspection PyUnusedLocal
    def training_printer(ses, step):
        print('training error', error.eval(feed_dict=val_and_train()),
              'training accuracy', accuracy.eval(feed_dict=tr_s()),
              'validation accuracy', accuracy.eval(feed_dict=val_s()))

    printer = PrintUtils(stepwise_pu(training_printer, 1000))

    validation_accuracies = []

    with tf.Session().as_default() as ss:

        for val_rho in rho_search_space:
            print('searching for rho', val_rho)
            tf.global_variables_initializer().run()

            for k in range(T):
                ss.run(ts, feed_dict=tr_s())
                printer.run(ss, k)

            svd = do_some_saves()
            print('validation accuracy', svd['validation accuracy'])

            validation_accuracies.append(svd['validation accuracy'])

        print('end')

        best_index = np.argmax(validation_accuracies)
        val_rho = rho_search_space[best_index]

        for k in range(T):  # last round, training on both training and validation set
            ss.run(ts, feed_dict=val_and_train())
            printer.run(ss, k)

        return do_some_saves(append_name='BEST')


main_experiment_available_strategies = {
    'default': None,
    'adam_and_val_only': None,
    'eta only': None,
}


def main_experiment(name_of_experiment, datasets,
                    T=3000, hyper_parameter_optimizer_strategy='default', hyper_iterations=250,
                    learning_rate=.01, momentum=.9, A_init=0, rho_init=0., collect_data=True,
                    coordinate_descent=False,
                    do_plots=False):

    if hyper_parameter_optimizer_strategy not in list(main_experiment_available_strategies.keys()):
        raise AttributeError('hyper_parameter_optimizer_strategy %s not implemented. Choose among %s'
                             % (hyper_parameter_optimizer_strategy, list(main_experiment_available_strategies.keys())))

    settings['NOTEBOOK_TITLE'] = name_of_experiment

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = LinearModel(x, datasets.train.dim_data, datasets.train.dim_target)

    res = vectorize_model(model.var_list, model.inp[-1], model.Ws[0], model.bs[0], augment=1)

    w = res[0]
    out = res[1]
    W_mod = res[2]
    b_mod = res[3]

    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    error = tf.reduce_mean(cross_entropy_loss(out, y))

    rho = tf.Variable(rho_init, name='rho')
    L2reg = rho*tf.reduce_sum(W_mod**2)

    if isinstance(A_init, float):  # assumes A_init is standard deviation of normal distribution
        A_init_value = tf.maximum(
            tf.abs(tf.random_normal([datasets.train.dim_target, datasets.train.dim_target], stddev=A_init)) -
            100.*tf.eye(datasets.train.dim_target), 0)
        A_init_value = (tf.transpose(A_init_value) + A_init_value)/2.  # symmetric matrix with 0 entries on diagonal
    else:
        A_init_value = A_init
    A = tf.Variable(A_init_value, dtype=tf.float32, name='A')

    vct_norm2 = tf.reduce_sum(W_mod**2, axis=[0])
    _t = tf.zeros([vct_norm2.get_shape().as_list()[0], vct_norm2.get_shape().as_list()[0]]) + vct_norm2
    # noinspection PyTypeChecker
    _tt = A*(-2*tf.matmul(tf.transpose(W_mod), W_mod) + tf.transpose(_t) + _t)
    multitask_reg = tf.reduce_sum(_tt)

    err_train = error + L2reg + multitask_reg

    eta = tf.Variable(learning_rate, name='eta')

    mu = tf.Variable(momentum, name='mu')

    dynamics_dict = momentum_dynamics(w, lr=eta, mu=mu, loss=err_train)

    val_err_dict = {
        'default': {error: [rho, A], err_train: [eta, mu]},
        'adam_and_val_only': {error: [rho, A, eta, mu]},
        'eta only': {err_train: eta}
    }

    doh = ReverseHyperGradient(dynamics_dict, hyper_dict=val_err_dict[hyper_parameter_optimizer_strategy])

    # noinspection PyUnusedLocal
    def tr_s(step=None): return {x: datasets.train.data, y: datasets.train.target}

    def val_and_train():
        return {x: np.vstack([datasets.train.data, datasets.validation.data]),
                y: np.vstack([datasets.train.target, datasets.validation.target])}

    # noinspection PyUnusedLocal
    def val_s(step=None): return {x: datasets.validation.data, y: datasets.validation.target}

    # noinspection PyUnusedLocal
    def tst_s(step=None): return {x: datasets.test.data, y: datasets.test.target}

    grad_hyper = tf.placeholder(tf.float32)

    def build_std_hyper_parameter_optimizers():
        adam_hlr = 1.e-3  # default setting for adam
        # gd_hlr = 1.e-2

        def get_adam(_hyp):
            return adam_dynamics(_hyp, lr=adam_hlr, grad=grad_hyper, w_is_state=False)

        # def get_gd(_hyp):
        #    return gradient_descent(_hyp, lr=gd_hlr, grad=grad_hyper)

        _opt_hyper_dicts = {A: get_adam(A), rho: get_adam(rho),
                            eta: get_adam(eta), mu: get_adam(mu)}
        positivity = {hyper: hyper.assign(tf.maximum(hyper, tf.zeros_like(hyper))) for hyper in doh.hyper_list}
        return _opt_hyper_dicts, positivity

    def build_adam_only():
        adam_hlr = 1.e-3  # default setting for adam

        def get_adam(_hyp):
            return adam_dynamics(_hyp, lr=adam_hlr, grad=grad_hyper, w_is_state=False)

        _opt_hyper_dicts = {A: get_adam(A), rho: get_adam(rho),
                            eta: get_adam(eta), mu: get_adam(mu)}
        positivity = {hyper: hyper.assign(tf.maximum(hyper, tf.zeros_like(hyper))) for hyper in doh.hyper_list}
        return _opt_hyper_dicts, positivity

    hyper_opt_options = {
        'default': build_std_hyper_parameter_optimizers(),
        'adam_and_val_only': build_adam_only(),
        'eta only': build_adam_only()
    }

    opt_hyper_dicts, simple_constraints = hyper_opt_options[hyper_parameter_optimizer_strategy]

    collected_hyper_gradients = {}  # DO NOT DELETE!
    validation_scores = []

    # noinspection PyUnusedLocal
    def do_some_saves(ses, step, append_string=""):
        nonlocal collected_hyper_gradients, hyper_it
        name = str(hyper_it) + append_string
        save_dict = {
            'test accuracy': accuracy.eval(feed_dict=tst_s()),
            'validation accuracy': accuracy.eval(feed_dict=val_s()),
            'training accuracy': accuracy.eval(feed_dict=tr_s()),
            'validation error': error.eval(feed_dict=val_s()),
            simple_name(A): A.eval(),
            simple_name(rho): rho.eval(),
            simple_name(eta): eta.eval(),
            simple_name(mu): mu.eval(),
            'W': W_mod.eval(),
            'b': b_mod.eval(),
            'previous hyper-gradient': {_hyp.name: grad for _hyp, grad in collected_hyper_gradients.items()}
        }
        if collect_data: save_obj(save_dict, name)
        validation_scores.append(save_dict['validation accuracy'])  # we choose to best w.r.t. validation accuracy
        return save_dict

    acc_printer = PrintUtils(unconditional_pu(lambda ses, step: print(
        'test accuracy', accuracy.eval(feed_dict=tst_s()),
        'validation accuracy', accuracy.eval(feed_dict=val_s()),
        'training', accuracy.eval(feed_dict=tr_s()),
        'validation error', error.eval(feed_dict=val_s()))),
                             unconditional_pu(do_some_saves))

    backward_pu = PrintUtils(stepwise_pu(
        lambda ses, step: print([np.linalg.norm(p.eval()) for p in doh.p_dict.values()]), 1000))

    forward_pu = PrintUtils(stepwise_pu(
        lambda ses, step: print('training error', err_train.eval(feed_dict=tr_s(step))), 1000))

    with tf.Session(config=config).as_default() as ss:
        tf.variables_initializer([eta, mu, rho, A]).run()  # all possible hyper-parameters

        [ahd.support_variables_initializer().run() for ahd in opt_hyper_dicts.values()]

        for hyper_it in range(hyper_iterations):
            print('hyperiteration', hyper_it)

            val_supp_possibility_dict = {
                'default': {error: val_s, err_train: tr_s},
                'adam_and_val_only': {error: val_s},
                'eta only': {err_train: tr_s}
            }

            res = doh.run_all(T, training_supplier=tr_s, backward_su=backward_pu, forward_su=forward_pu,
                              validation_suppliers=val_supp_possibility_dict[hyper_parameter_optimizer_strategy],
                              after_forward_su=acc_printer)

            collected_hyper_gradients = ReverseHyperGradient.std_collect_hyper_gradients(res)

            print('hyper gradients')
            print_hyper_gradients(collected_hyper_gradients)

            print()

            print([np.linalg.norm(c) for c in collected_hyper_gradients.values()])

            update_list = [doh.hyper_list[hyper_it % len(doh.hyper_list)]] if coordinate_descent else doh.hyper_list

            for hyp in update_list:
                ss.run(opt_hyper_dicts[hyp].assign_ops, feed_dict={grad_hyper: collected_hyper_gradients[hyp]})
                opt_hyper_dicts[hyp].increase_global_step()
                ss.run(simple_constraints[hyp])

            if do_plots:
                plt.matshow(A.eval())
                plt.grid()
                plt.colorbar()
                plt.show()

            print('rho, eta, mu', ss.run([rho, eta, mu]))
            print('norm A', np.linalg.norm(ss.run(A)))

            print()

        best_index = np.argmax(validation_scores)
        res = load_obj(str(best_index))  # same naming convention for the files in do_some_saves

        for hyp in doh.hyper_list:  # sets the values of the hyper-parameters to the best ones
            hyp.assign(x).eval(feed_dict={x: res[simple_name(hyp)]})  # is it necessary or can I use feed_dict???

        tf.variables_initializer(w.var_list(Vl_Mode.BASE))
        for k in range(T):
            ss.run(dynamics_dict.assign_ops, feed_dict=val_and_train())

        hyper_it = 'BEST'

        return do_some_saves(ss, -1, append_string='at_hyper_iteration_%d' % best_index)


def naive_multitask(name_of_experiment, datasets, T=3000,
                    learning_rate=.01, momentum=.9,
                    rho_search_space=None,
                    a_search_space=None,
                    collect_data=True):

    # Train - validation - test
    train_x, val_x, test_x = datasets.train.data, datasets.validation.data, datasets.test.data
    train_y, val_y, test_y = datasets.train.target, datasets.validation.target, datasets.test.target

    settings['NOTEBOOK_TITLE'] = name_of_experiment

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = LinearModel(x, len(train_x[1][:]), datasets.train.dim_target)

    out, W_mod, b_mod = (model.inp[-1], model.Ws[0], model.bs[0])
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    error = tf.reduce_mean(cross_entropy_loss(out, y))

    rho = tf.placeholder(tf.float32, name='rho')
    L2reg = rho*tf.reduce_sum(W_mod**2)

    ones = tf.ones([datasets.train.dim_target, datasets.train.dim_target])

    a = tf.placeholder(tf.float32, name='A')

    A = a*ones

    vct_norm2 = tf.reduce_sum(W_mod**2, axis=[0])
    _t = tf.zeros([vct_norm2.get_shape().as_list()[0], vct_norm2.get_shape().as_list()[0]]) + vct_norm2
    # noinspection PyTypeChecker
    _tt = A*(-2*tf.matmul(tf.transpose(W_mod), W_mod) + tf.transpose(_t) + _t)
    multitask_reg = tf.reduce_sum(_tt)

    err_train = error + L2reg + multitask_reg

    # noinspection PyUnusedLocal
    def tr_s(step=None): return {x: train_x, y: train_y, a: val_a, rho: val_rho}

    def val_and_train():
        return {x: np.vstack([train_x, val_x]), y: np.vstack([train_y, val_y]), rho: val_rho, a: val_a}

    # noinspection PyUnusedLocal
    def val_s(step=None): return {x: val_x, y: val_y}

    # noinspection PyUnusedLocal
    def tst_s(step=None): return {x: test_x, y: test_y}

    def do_some_saves(append_string=""):
        nonlocal val_rho, val_a
        name = 'a_%.4e_rho_%.4e%s' % (val_rho, val_a, append_string)
        save_dict = {
            'test accuracy': accuracy.eval(feed_dict=tst_s()),
            'validation accuracy': accuracy.eval(feed_dict=val_s()),
            'training accuracy': accuracy.eval(feed_dict=tr_s()),
            'A': val_a,
            'rho': val_rho,
            'W': W_mod.eval(),
            'b': b_mod.eval()
        }
        if collect_data: save_obj(save_dict, name)
        return save_dict

    acc_printer = PrintUtils(unconditional_pu(lambda ses, step: print(
        'test accuracy', accuracy.eval(feed_dict=tst_s()),
        'validation accuracy', accuracy.eval(feed_dict=val_s()),
        'training', accuracy.eval(feed_dict=tr_s()),
        'validation error', error.eval(feed_dict=val_s()))))

    forward_pu = PrintUtils(stepwise_pu(
        lambda ses, step: print('training error', err_train.eval(feed_dict=tr_s(step))), 1000))

    ts = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)\
        .minimize(err_train, var_list=model.var_list)

    validation_scores = []
    with tf.Session(config=config).as_default() as ss:

        for val_rho in rho_search_space:
            for val_a in a_search_space:

                tf.global_variables_initializer().run()
                for k in range(T):
                    ss.run(ts, feed_dict=tr_s())
                    forward_pu.run(ss, k)
                acc_printer.run()
                res = do_some_saves()
                validation_scores.append(res['training accuracy'])
                print()
        print('end')

        best_index = np.argmax(validation_scores)
        val_rho, val_a = (rho_search_space[best_index], a_search_space[best_index])
        print('best roh, best a:', val_rho, val_a)

        tf.global_variables_initializer().run()
        for k in range(T):
            ss.run(ts, feed_dict=val_and_train())
            forward_pu.run(ss, k)
        acc_printer.run()
        return do_some_saves('BEST')
