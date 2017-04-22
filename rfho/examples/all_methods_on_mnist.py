"""
This module contains a set of example of the execution of the three main algorithms contained in this package:
- Reverse-HO
- Forward-HO and
- RealTimeHO (RTHO)
 on various classifiers trained on MNIST dataset, with different hyperparameter settings.
"""
import tensorflow as tf
import rfho as rf
import numpy as np


def load_dataset(partition_proportions=(.5, .3)):
    from rfho.datasets import load_mnist
    return load_mnist(partitions=partition_proportions)


IMPLEMENTED_MODEL_TYPES = ['log_reg', 'ffnn']
HO_MODES = ['forward', 'reverse', 'rtho']

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def create_model(datasets, model_creator='log_reg', **model_kwargs):
    dataset = datasets.train
    x = tf.placeholder(tf.float32)
    assert model_creator in IMPLEMENTED_MODEL_TYPES or callable(model_creator), \
        '%s, available %s \n You can implement your own model' \
        'with a function that returns the model! Since a certain structure is assumed ' \
        'in the model object to proceed automatically with the vectorization,' \
        'your model should be a subclass of rf.models.Network...' % (model_creator, IMPLEMENTED_MODEL_TYPES)
    if model_creator == IMPLEMENTED_MODEL_TYPES[0]:
        model = create_logistic_regressor(x, (dataset.dim_data, dataset.dim_target), **model_kwargs)
    elif model_creator == IMPLEMENTED_MODEL_TYPES[1]:  # ffnn deep
        model = create_ffnn(x, dataset.dim_data, dataset.dim_target, **model_kwargs)
    else:  # custom _model creator
        model = model_creator(x, **model_kwargs)

    return x, model


def create_logistic_regressor(x, dimensions, **model_kwargs):
    return rf.LinearModel(x, dimensions[0], dimensions[1], **model_kwargs)


def create_ffnn(x, d0, d1, **model_kwargs):
    dimensions = model_kwargs.get('dims', None)
    if dimensions is None: dimensions = [None, 50, 50, 50, 50, 50, None]  # like in MacLaurin (maybe deeper)
    dimensions[0], dimensions[-1] = d0, d1
    model_kwargs['dims'] = dimensions
    return rf.FFNN(x, **model_kwargs)


def define_errors_default_models(model, l1=0., l2=0., synthetic_hypers=None, augment=0):
    assert isinstance(model, rf.Network)

    res = rf.vectorize_model(model.var_list, model.inp[-1], *model.Ws,
                             augment=augment)
    s, out, ws = res[0], res[1], res[2:]

    # error
    y = tf.placeholder(tf.float32)
    error = tf.reduce_mean(rf.cross_entropy_loss(out, y))  # also validation error

    base_training_error = rf.cross_entropy_loss(out, y)

    gamma = None
    if synthetic_hypers is not None:
        gamma = tf.Variable(tf.ones([synthetic_hypers]))
        training_error = tf.reduce_mean([gamma[k]*base_training_error[k] for k in range(synthetic_hypers)])
    else:
        training_error = tf.reduce_mean(base_training_error)

    rho_l1s, reg_l1s, rho_l2s, reg_l2s = None, None, None, None

    # layer-wise l1 regularizers]
    if isinstance(l1, float):
        rho_l1s = [tf.Variable(l1, name='rho_l1_%d' % k) for k in range(len(ws))]
        reg_l1s = [tf.reduce_sum(tf.abs(w)) for w in ws]
        training_error += tf.reduce_sum([rho*rg_l1 for rho, rg_l1 in zip(rho_l1s, reg_l1s)])

    # layer-wise l2 regularizers]
    if isinstance(l2, float):
        rho_l2s = [tf.Variable(l1, name='rho_l2_%d' % k) for k in range(len(ws))]
        reg_l2s = [tf.reduce_sum(tf.pow(w, 2)) for w in ws]
        training_error += tf.reduce_sum([rho * rg_l1 for rho, rg_l1 in zip(rho_l2s, reg_l2s)])

    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return s, out, ws, y, error, training_error, rho_l1s, reg_l1s, rho_l2s, reg_l2s, accuracy,\
           base_training_error, gamma


def experiment(name_of_experiment, collect_data=True,
               datasets=None, model='log_reg', model_kwargs=None, l1=0., l2=0.,
               synthetic_hypers=None, set_T=None,
               optimizer=rf.MomentumOptimizer, optimizer_kwargs=None, batch_size=200,
               algo_hyper_wrt_tr_error=False,
               mode='reverse', hyper_optimizer=rf.AdamOptimizer, hyper_optimizer_kwargs=None,
               hyper_iterations=100, hyper_batch_size=100, epochs=20, do_print=True):

    assert mode in HO_MODES

    # set random seeds!!!!
    np.random.seed(1)
    tf.set_random_seed(1)

    if synthetic_hypers is not None:
        batch_size = synthetic_hypers  # altrimenti divento matto!

    from rfho.examples.common import save_setting, Saver
    if name_of_experiment is not None: rf.settings['NOTEBOOK_TITLE'] = name_of_experiment
    if collect_data: save_setting(vars(), excluded=datasets, append_string='_%s' %mode)

    if datasets is None: datasets = load_dataset()

    x, model = create_model(datasets, model, **model_kwargs or {})
    s, out, ws, y, error, training_error, rho_l1s, reg_l1s, rho_l2s, reg_l2s, accuracy, base_tr_error, gamma = \
        define_errors_default_models(model, l1, l2, synthetic_hypers=synthetic_hypers,
                                     augment=optimizer.get_augmentation_multiplier())

    if optimizer_kwargs is None: optimizer_kwargs = {'lr': tf.Variable(.01, name='eta'),
                                                     'mu': tf.Variable(.5, name='mu')}

    tr_dynamics = optimizer.create(s, loss=training_error, w_is_state=True, **optimizer_kwargs)

    # hyperparameters part!
    algorithmic_hyperparameters = []
    eta = tr_dynamics.learning_rate
    if isinstance(eta, tf.Variable):
        if mode != 'reverse':
            algorithmic_hyperparameters.append((eta, tr_dynamics.d_dynamics_d_learning_rate()))
        else:
            algorithmic_hyperparameters.append(eta)
    if hasattr(tr_dynamics, 'momentum_factor'):
        mu = tr_dynamics.momentum_factor
        if isinstance(mu, tf.Variable):
            if mode != 'reverse':
                algorithmic_hyperparameters.append((mu, tr_dynamics.d_dynamics_d_momentum_factor()))
            else:
                algorithmic_hyperparameters.append(mu)

    regularization_hyperparameters = []
    vec_w = s.var_list(rf.Vl_Mode.TENSOR)[0]  # vectorized representation of _model weights (always the first!)
    if rho_l1s is not None:
        if mode != 'reverse':
            regularization_hyperparameters += [(r1, tr_dynamics.d_dynamics_d_linear_loss_term(
                tf.gradients(er1, vec_w)[0])) for r1, er1 in zip(rho_l1s, reg_l1s)]
        else: regularization_hyperparameters += rho_l1s
    if rho_l2s is not None:
        if mode != 'reverse':
            regularization_hyperparameters += [(r2, tr_dynamics.d_dynamics_d_linear_loss_term(
                tf.gradients(er2, vec_w)[0])) for r2, er2 in zip(rho_l2s, reg_l2s)]
        else: regularization_hyperparameters += rho_l2s

    synthetic_hyperparameters = []
    if synthetic_hypers:
        if mode != 'reverse':
            da_grad = tf.transpose(tf.stack(
                [tf.gradients(base_tr_error[k], vec_w)[0] for k in range(synthetic_hypers)]
            ))

            d_phi_d_gamma = rf.utils.ZMergedMatrix([
                - eta * da_grad,
                da_grad
            ])
            synthetic_hyperparameters.append(
                (gamma, d_phi_d_gamma)
            )
        else:
            synthetic_hyperparameters.append(gamma)

    # end of hyperparameters

    # create hyper_dict
    hyper_dict = {error: regularization_hyperparameters + synthetic_hyperparameters}
    if algo_hyper_wrt_tr_error:  # it is possible to optimize different hyperparameters wrt different validation errors
        hyper_dict[training_error] = algorithmic_hyperparameters
    else: hyper_dict[error] += algorithmic_hyperparameters

    print(hyper_dict)

    hyper_gradients = rf.ReverseHyperGradient(tr_dynamics, hyper_dict) if mode == 'reverse' else \
        rf.ForwardHyperGradient(tr_dynamics, hyper_dict)

    hyper_optimizers = rf.create_hyperparameter_optimizers(hyper_gradients, hyper_optimizer,
                                                           **hyper_optimizer_kwargs or {})
    positivity = rf.positivity(hyper_gradients.hyper_list)

    # builds an instance of Real Time Hyperparameter optimization if mode is rtho
    # RealTimeHO exploits partial hypergradients calculated with forward-mode to perform hyperparameter updates
    # while the _model is training...
    rtho = rf.RealTimeHO(hyper_gradients, hyper_optimizers, positivity) if mode == 'rtho' else None

    # stochastic descent
    import rfho.datasets as dt
    ev_data = dt.ExampleVisiting(datasets, batch_size=batch_size, epochs=epochs)
    ev_data.generate_visiting_scheme()
    tr_supplier = ev_data.create_train_feed_dict_supplier(x, y)
    val_supplier = ev_data.create_all_valid_feed_dict_supplier(x, y)
    test_supplier = ev_data.create_all_test_feed_dict_supplier(x, y)

    def all_training_supplier(step=None):
        return {x: datasets.train.data, y: datasets.train.target}

    # feed_dict supplier for validation errors
    val_feed_dict_suppliers = {error: val_supplier}
    if algo_hyper_wrt_tr_error: val_feed_dict_suppliers[training_error] = all_training_supplier

    def calculate_memory_usage():
        memory_usage = rf.simple_size_of_with_pickle([
            hyper_gradients.w.eval(),
            [h.eval() for h in hyper_gradients.hyper_list]
        ])
        if mode == 'reverse':
            return memory_usage + rf.simple_size_of_with_pickle([
                hyper_gradients.w_hist,
                [p.eval() for p in hyper_gradients.p_dict.values()]
            ])
        else:
            return memory_usage + rf.simple_size_of_with_pickle([
                [z.eval() for z in hyper_gradients.zs]
            ])

    # number of iterations
    T = set_T or ev_data.T

    hyper_grads = hyper_gradients.hyper_gradients_dict
    # create a Saver object
    saver = Saver(
        'step', lambda step: step,
        'test accuracy', accuracy, test_supplier,
        'validation accuracy', accuracy,val_supplier,
        'training accuracy', accuracy, tr_supplier,
        'validation error', error, val_supplier,
        'memory usage (mb)', lambda step: calculate_memory_usage()*9.5367e-7,
        'weights', vec_w,
        '# weights', lambda step: vec_w.get_shape().as_list()[0],
        '# hyperparameters', lambda step: len(hyper_gradients.hyper_list),
        '# iterations', lambda step: T,
        *rf.flatten_list([rf.simple_name(hyp), [hyp, hyper_grads[hyp]]]
                         for hyp in hyper_gradients.hyper_list),
        do_print=do_print, collect_data=collect_data
    )

    with tf.Session(config=config).as_default() as ss:
        saver.timer.start()

        if mode == 'rtho':  # here we do not have hyper-iterations
            rtho.initialize()  # helper for initializing all variables...
            for k in range(hyper_iterations):
                rtho.hyper_batch(hyper_batch_size, train_feed_dict_supplier=tr_supplier,
                                 val_feed_dict_suppliers=val_feed_dict_suppliers)

                saver.save(k, append_string='_%s' % mode)

        else:  # here we do complete hyper-iterations..
            #  initialize hyperparameters and support variables of hyperparameter optimizers
            tf.variables_initializer(hyper_gradients.hyper_list).run()
            [hod.support_variables_initializer().run() for hod in hyper_optimizers]

            for k in range(hyper_iterations):  # start hyper-iterations
                hyper_gradients.run_all(T, train_feed_dict_supplier=tr_supplier,
                                        val_feed_dict_suppliers=val_feed_dict_suppliers)

                # update hyperparameters
                [ss.run(hod.assign_ops) for hod in hyper_optimizers]
                [ss.run(prj) for prj in positivity]

                saver.save(k, append_string='_%s' % mode)


if __name__ == '__main__':
    synt_hyp = None
    for _mode in HO_MODES:
        for _model in IMPLEMENTED_MODEL_TYPES[1:2]:
            _model_kwargs = {'dims': [None, 200, 200, 200, None]}
            tf.reset_default_graph()
            experiment('test_with_model_' + _model, collect_data=False, hyper_iterations=2, mode=_mode, epochs=3,
                       model=_model,
                       model_kwargs=_model_kwargs,
                       set_T=100,
                       synthetic_hypers=synt_hyp
                       # optimizer=rf.GradientDescentOptimizer,
                       # optimizer_kwargs={'lr': tf.Variable(.01, name='eta')}
                       )