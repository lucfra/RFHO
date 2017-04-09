import numpy as np
import tensorflow as tf

from time import time

from copy import deepcopy

from rfho.datasets import ExampleVisiting
from rfho.hyper_gradients import ForwardHyperGradient
from rfho.optimizers import momentum_dynamics, adam_dynamics
from rfho.models import vectorize_model, FFNN, ffnn_layer
from rfho.save_and_load import save_obj, settings
from rfho.utils import cross_entropy_loss, stepwise_pu, PrintUtils, Vl_Mode, simple_name, ZMergedMatrix
from rfho.experiments.common import save_setting

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

available_strategies = {
    'default': None,
    'alg hyper on training': None,
}


def main_experiment(name_of_experiment, ev_data,
                    momentum=.5, learning_rate=.075, lambda_0=0., gamma_0=None,
                    hidden_units=400,
                    hyper_learning_rate=5.e-3, hyper_iterations=100,
                    online_hyper_update=False, as_stream=False, validation_samples=100000,
                    collect_data=True):
    assert isinstance(ev_data, ExampleVisiting), 'NOT IMPLEMENTED. :('

    settings['NOTEBOOK_TITLE'] = name_of_experiment
    if collect_data: save_setting(vars())

    datasets = ev_data.datasets

    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')

        model = FFNN(x, [datasets.train.dim_data, hidden_units, hidden_units, hidden_units,
                         hidden_units, datasets.train.dim_target], activ_gen=(
            ffnn_layer(init_w=tf.contrib.layers.xavier_initializer(),
                       activ=tf.nn.relu),
            ffnn_layer(activ=lambda features, name: tf.concat([  # NOTE 8/03/17: in previous versions, due to a mistake
                # the ReLu was not applied to the regression part of the output. This was a minor mistake
                # since applying the ReLu is a design choice, yet we should run again all the experiments
                # Note also that we should discard the components of the embedding vectors that are always 0
                # (should be around 150...)
                features[:, :datasets.validation.dim_target], tf.nn.relu(features[:, datasets.validation.dim_target:])
            ], 1))))

        w, out, w_inner1, w_inner2, w_inner3, w_inner4 = vectorize_model(model.var_list, model.inp[-1],
                                                                         model.Ws[0], model.Ws[1], model.Ws[2],
                                                                         model.Ws[3], augment=1)

        primary_out = out[:, :datasets.validation.dim_target]
        secondary_out = out[:, datasets.validation.dim_target:]

        primary_error = tf.reduce_mean(cross_entropy_loss(primary_out, y[:, :datasets.validation.dim_target]))

        lambda_secondary = tf.Variable(lambda_0, name='lambda')
        secondary_error = tf.reduce_mean((secondary_out - y[:, datasets.test.dim_target:])**2)/2.

        l2_reg = tf.reduce_sum(w.tensor**2)

        # noinspection PyTypeChecker
        training_error = primary_error + lambda_secondary*secondary_error  # + gamma*l2_reg

        gamma = 0.
        if gamma_0:
            gamma = tf.Variable(gamma_0, name='gamma')
            training_error += gamma*l2_reg

        correct_prediction = tf.equal(tf.argmax(primary_out, 1), tf.argmax(y[:, :datasets.test.dim_target], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        eta = tf.Variable(learning_rate, name='eta')
        mu = tf.Variable(momentum, name='mu')

        dynamics_dict = momentum_dynamics(w, lr=eta, mu=mu, loss=training_error)

        w_b, m = w.var_list(Vl_Mode.TENSOR)
        grad_E = tf.gradients(training_error, w_b)[0]
        grad_S = tf.gradients(secondary_error, w_b)[0]
        grad_reg = tf.gradients(l2_reg, w_b)[0]

        d_eta = ZMergedMatrix([
            - tf.transpose([mu * m + grad_E]),
            tf.zeros([m.get_shape().as_list()[0], 1])
        ])
        d_mu = ZMergedMatrix([
            - (eta * m),
            m
        ])

        d_gamma = ZMergedMatrix([
            - (eta*grad_reg),
            grad_reg
        ])

        d_lambda = ZMergedMatrix([
            - (eta * grad_S),
            grad_S
        ])

        hyper_dict = {
            # training_error: [(eta, d_eta), (mu, d_mu)],
            primary_error: [(eta, d_eta), (mu, d_mu), (lambda_secondary, d_lambda)]
        }

        if gamma_0:
            hyper_dict[primary_error].append((gamma, d_gamma))

        direct_doh = ForwardHyperGradient(optimizer=dynamics_dict, hyper_dict=hyper_dict)

        training_supplier = ev_data.training_supplier(x, y)

        # noinspection PyUnusedLocal
        def validation_supplier(step=None):
            random_samples = np.random.choice(list(range(datasets.validation.num_examples)), size=(validation_samples,),
                                              replace=False)
            return {x: datasets.validation.data[random_samples, :], y: datasets.validation.target[random_samples, :]}

        # noinspection PyUnusedLocal
        def training_val_supplier(step=None):
            random_samples = np.random.choice(list(range(datasets.train.num_examples)), size=(validation_samples,),
                                              replace=False)
            return {x: datasets.train.data[random_samples, :], y: datasets.train.target[random_samples, :]}

        test_supplier = ev_data.all_test_supplier(x, y)

        def forward_printer(sss, step):
            print(step, 'test accuracy:', accuracy.eval(feed_dict=test_supplier()),
                  '[primary error, secondary error]',
                  sss.run([primary_error, secondary_error], feed_dict=training_supplier(step)))

        norm_zs = [tf.sqrt(tf.reduce_sum(z.components[0]**2 + z.components[1]**2)) for z in direct_doh.zs]

        # noinspection PyUnusedLocal
        def z_printer(sss, step):
            print('norm_z relative to hyper_parameters')
            print(*[simple_name(hy) for hy in direct_doh.hyper_list], sep='\t')
            print(*sss.run(norm_zs), sep='\t')

        fsu = PrintUtils(stepwise_pu(forward_printer, 2500), stepwise_pu(z_printer, 2500))

        # collected_hyper_gradients = {}  # should not be needed here.
        starting_time = time()

        # noinspection PyUnusedLocal
        def do_some_saves(ses, step, append_string=""):
            nonlocal collected_hyper_gradients
            name = str(step) + append_string
            save_dict = {
                'time': time() - starting_time,
                'test accuracy': accuracy.eval(feed_dict=test_supplier()),
                'validation accuracy': accuracy.eval(feed_dict=validation_supplier()),
                'training accuracy': accuracy.eval(feed_dict=training_supplier(
                    np.random.randint(0, len(ev_data.training_schedule)))),
                'validation error': primary_error.eval(feed_dict=validation_supplier()),
                simple_name(lambda_secondary): lambda_secondary.eval(),
                simple_name(eta): eta.eval(),
                simple_name(mu): mu.eval(),
                'hyper-gradients': {simple_name(hyp): grd for hyp, grd in collected_hyper_gradients.items()}
            }
            if gamma_0:
                save_dict[simple_name(gamma)] = gamma.eval()
            for key, v in save_dict.items(): print(key, v)
            if collect_data: save_obj(save_dict, name)

            return save_dict

        grad_hyper = tf.placeholder(tf.float32)
        hyper_opt_dicts = {hyp: adam_dynamics(hyp, lr=hyper_learning_rate,
                                              grad=grad_hyper, w_is_state=False) for hyp in direct_doh.hyper_list}
        positivity = {hyp: hyp.assign(tf.maximum(hyp, tf.zeros_like(hyp))) for hyp in direct_doh.hyper_list}

        if not online_hyper_update: online_hyper_update = ev_data.T - 1

        with tf.Session(config=config).as_default() as ss:
            tf.variables_initializer(direct_doh.hyper_list).run()

            [ahd.support_variables_initializer().run() for ahd in hyper_opt_dicts.values()]

            for hyper_it in range(hyper_iterations*(1 - as_stream) + 1):  # just do an hyper-iteration if is stream
                print('Hyper-iteration', hyper_it + 1)

                # if hyper_it == 0 or not as_stream:
                direct_doh.initialize()  # if is a stream do not reinitialize
                # if as_stream:
                #    ev_data.generate_visiting_scheme()

                for k in range(ev_data.T*(1 + as_stream*hyper_iterations)):

                    direct_doh.step_forward(feed_dict_supplier=training_supplier)
                    fsu.run(ss, k)

                    if k != 0 and k % online_hyper_update == 0:
                        print('optimizing hyper_parameters at step %d' % k)

                        collected_hyper_gradients = direct_doh.hyper_gradient_vars(
                            validation_suppliers={training_error: training_val_supplier,
                                                  primary_error: validation_supplier})

                        # print_hyper_gradients(collected_hyper_gradients)  # quite redundant as a print...
                        for hyp in direct_doh.hyper_list:
                            ss.run(hyper_opt_dicts[hyp].assign_ops,
                                   feed_dict={grad_hyper: collected_hyper_gradients[hyp]})
                            ss.run(positivity[hyp])
                            hyper_opt_dicts[hyp].increase_global_step()

                        do_some_saves(ss, hyper_it, append_string='_update_%d' % (k/online_hyper_update))
                        print()

                print('end of optimization: ')
                if online_hyper_update != ev_data.T - 1:
                    do_some_saves(ss, hyper_it, append_string='_last')
                print()

            return do_some_saves(ss, 'LAST')


def random_search(name_of_experiment, ev_data, sampler,
                  search_time=18000,
                  collect_data=True, hidden_units=400):
    start = time()
    trial = 0
    results = []
    while time() - start < search_time:
        print('BEGIN TRIAL %d' % trial)
        eta, mu, lambda_0 = sampler()
        ev_data.generate_visiting_scheme()
        res, _ = baseline(name_of_experiment + str(trial), ev_data, learning_rate=eta, momentum=mu,
                          lambda_search_space=lambda_0, hidden_units=hidden_units, collect_data=collect_data)
        results.append(res)
        print()
        trial += 1

    settings['NOTEBOOK_TITLE'] = name_of_experiment
    if collect_data: save_obj(results, 'results')
    return results


def baseline(name_of_experiment, ev_data,
             momentum=.5, learning_rate=.075, lambda_search_space=np.array([0.]),
             hidden_units=400, initial_patience=10,
             collect_data=True, acceptance_factor=.01):
    print('running baseline with early stopping')
    assert isinstance(ev_data, ExampleVisiting), 'NOT IMPLEMENTED. :('

    if isinstance(lambda_search_space, float): lambda_search_space = np.array([lambda_search_space])

    settings['NOTEBOOK_TITLE'] = name_of_experiment
    if collect_data: save_setting(vars())

    datasets = ev_data.datasets

    with tf.Graph().as_default():

        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')

        model = FFNN(x, [datasets.train.dim_data, hidden_units, hidden_units, hidden_units,
                         hidden_units, datasets.train.dim_target], activ_gen=(
            ffnn_layer(init_w=tf.contrib.layers.xavier_initializer(),
                       activ=tf.nn.relu),
            ffnn_layer(activ=lambda features, name: tf.concat([
                features[:, :datasets.train.dim_target], tf.nn.relu(features[:, datasets.train.dim_target:])
            ], 1))))

        out = model.inp[-1]

        primary_out = out[:, :datasets.test.dim_target]
        secondary_out = out[:, datasets.test.dim_target:]

        primary_error = tf.reduce_mean(cross_entropy_loss(primary_out, y[:, :datasets.test.dim_target]))

        lambda_secondary = tf.Variable(0., name='lambda')
        secondary_error = tf.reduce_mean((secondary_out - y[:, datasets.test.dim_target:])**2)/2.

        # l2_reg = (tf.reduce_sum(w_inner1**2) + tf.reduce_sum(w_inner2**2) + tf.reduce_sum(w_inner3**2)
        #           + tf.reduce_sum(w_inner4**2))
        # noinspection PyTypeChecker
        training_error = primary_error + lambda_secondary*secondary_error  # + gamma*l2_reg

        correct_prediction = tf.equal(tf.argmax(primary_out, 1), tf.argmax(y[:, :datasets.test.dim_target], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        eta = tf.Variable(learning_rate, name='eta')
        mu = tf.Variable(momentum, name='mu')

        ts = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                        momentum=momentum).minimize(loss=training_error, var_list=model.var_list)

        training_supplier = ev_data.training_supplier(x, y)

        # noinspection PyUnusedLocal
        def validation_supplier(step=None):
            random_samples = np.random.choice(list(range(datasets.validation.num_examples)), size=(100000,),
                                              replace=False)
            return {x: datasets.validation.data[random_samples, :], y: datasets.validation.target[random_samples, :]}

        # noinspection PyUnusedLocal
        def training_val_supplier(step=None):
            random_samples = np.random.choice(list(range(datasets.train.num_examples)), size=(100000,),
                                              replace=False)
            return {x: datasets.train.data[random_samples, :], y: datasets.train.target[random_samples, :]}

        test_supplier = ev_data.all_test_supplier(x, y)

        def forward_printer(sss, step):
            print(step, 'test accuracy:', accuracy.eval(feed_dict=test_supplier()),
                  '[primary error, secondary error]',
                  sss.run([primary_error, secondary_error], feed_dict=training_supplier(step)))

        fsu = PrintUtils(stepwise_pu(forward_printer, 2500))

        starting_time = time()

        # noinspection PyUnusedLocal
        def do_some_saves(ses, step, append_string="", overwrite=False):
            save_step = step
            if append_string == 'BEST':
                step = ''  # TODO fix this hack!
            name = str(step) + append_string
            save_dict = {
                'time': time() - starting_time,
                'step': step,
                'test accuracy': accuracy.eval(feed_dict=test_supplier()),
                'validation accuracy': accuracy.eval(feed_dict=validation_supplier()),
                'training accuracy': accuracy.eval(feed_dict=training_supplier(
                    np.random.randint(0, len(ev_data.training_schedule)))),
                'validation error': primary_error.eval(feed_dict=validation_supplier()),
                simple_name(lambda_secondary): lambda_secondary.eval(),
                simple_name(eta): eta.eval(),
                simple_name(mu): mu.eval(),
            }
            for key, v in save_dict.items(): print(key, v)
            if collect_data: save_obj(save_dict, name, default_overwrite=True)
            return save_dict

        best = None
        patience = initial_patience
        k = 0
        best_validation = 0.

        with tf.Session(config=config).as_default() as ss:
            tf.global_variables_initializer().run()

            for val_lambda in lambda_search_space:
                lambda_secondary.assign(x).eval(feed_dict={x: val_lambda})

                tf.variables_initializer(model.var_list).run()

                while k < ev_data.T and patience > 0:
                    ss.run(ts, feed_dict=training_supplier(k))
                    fsu.run(ss, k)

                    if k % 200 == 0 and k != 0:
                        print('_update_%d' % (k/200))

                        res = do_some_saves(ss, val_lambda, append_string='_update_%d' % (k/200))
                        if res['validation accuracy'] > best_validation*(1. + acceptance_factor):
                            best_validation = res['validation accuracy']
                            best = deepcopy(res)
                            patience = initial_patience
                        else: patience -= 1
                        print('patience', patience)
                        print()
                    k += 1

            return best, do_some_saves(ss, 0, append_string='discard')['time']
