"""
This module contains a simple example of the execution of the three algorithms on various classifiers
trained on mnist
"""
import tensorflow as tf
import rfho as rf


def load_dataset(partition_proportions=(.5, .3)):
    from rfho.datasets import load_mnist
    return load_mnist(partitions=partition_proportions)


_IMPLEMENTED_MODEL_TYPES = ['log_reg', 'ffnn']


def create_model(datasets, model_creator='log_reg', **model_kwargs):
    dataset = datasets.train
    x = tf.placeholder(tf.float32)
    assert model_creator in _IMPLEMENTED_MODEL_TYPES or callable(model_creator)
    if model_creator == _IMPLEMENTED_MODEL_TYPES[0]:
        model = create_logistic_regressor(x, (dataset.dim_data, dataset.dim_target), **model_kwargs)
    elif model_creator == _IMPLEMENTED_MODEL_TYPES[1]:
        dimensions = model_kwargs.get('dims', None)
        if dimensions is None: dimensions = [None, 50, 50, 50, None]  # like in MacLaurin
        dimensions[0], dimensions[-1] = dataset.dim_data, dataset.dim_target
        model = create_ffnn(x, dimensions, **model_kwargs)
    else:  # custom model creator
        model = model_creator(x, **model_kwargs)

    return x, model


def create_logistic_regressor(x, dimensions, **model_kwargs):
    return rf.LinearModel(x, dimensions[0], dimensions[1], **model_kwargs)


def create_ffnn(x, dimensions, **model_kwargs):
    return rf.FFNN(x, dimensions, **model_kwargs)


def define_errors_default_models(model, l1=0., l2=0., augment=0):
    assert isinstance(model, rf.Network)

    res = rf.vectorize_model(model.var_list, model.inp[-1], *model.Ws,
                             augment=augment)
    s, out, ws = res[0], res[1], res[2:]

    # error
    y = tf.placeholder(tf.float32)
    error = rf.cross_entropy_loss(out, y)  # also validation error
    training_error = error

    rho_l1s, reg_l1s, rho_l2s, reg_l2s = None, None, None, None

    # layer-wise l1 regularizers]
    if isinstance(l1, float):
        rho_l1s = [tf.Variable(l1, name='rho_l1_%d' % k) for k in range(len(ws))]
        reg_l1s = [tf.abs(w) for w in ws]
        training_error += tf.reduce_sum([rho*rg_l1 for rho, rg_l1 in zip(rho_l1s, reg_l1s)])

    # layer-wise l2 regularizers]
    if isinstance(l2, float):
        rho_l2s = [tf.Variable(l1, name='rho_l2_%d' % k) for k in range(len(ws))]
        reg_l2s = [tf.pow(w, 2) for w in ws]
        training_error += tf.reduce_sum([rho * rg_l1 for rho, rg_l1 in zip(rho_l2s, reg_l2s)])

    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return s, out, ws, y, error, training_error, rho_l1s, reg_l1s, rho_l2s, reg_l2s, accuracy

_HO_MODES = ['forward', 'reverse', 'rtho']


def experiment(name_of_experiment, collect_data=True,
               datasets=None, model='log_reg', model_kwargs=None, l1=0., l2=0.,
               optimizer=rf.MomentumOptimizer, optimizer_kwargs=None, batch_size=200,
               algo_hyper_wrt_tr_error=False,
               mode='reverse', hyper_optimizer=rf.AdamOptimizer, hyper_optimizer_kwargs=None,
               hyper_iterations=100, hyper_batch_size=100, epochs=20, do_print=True):
    assert mode in _HO_MODES

    from rfho.examples.common import save_setting, Saver
    if name_of_experiment is not None: rf.settings['NOTEBOOK_TITLE'] = name_of_experiment
    if collect_data: save_setting(vars(), excluded=datasets)

    if datasets is None: datasets = load_dataset()

    x, model = create_model(datasets, model, **model_kwargs or {})
    s, out, ws, y, error, training_error, rho_l1s, reg_l1s, rho_l2s, reg_l2s, accuracy = \
        define_errors_default_models(model, l1, l2, augment=optimizer.get_augmentation_multiplier())

    if optimizer_kwargs is None: optimizer_kwargs = {'lr': tf.Variable(.01, name='eta'),
                                                     'mu': tf.Variable(.5, name='mu')}

    tr_dynamics = optimizer.create(s, loss=training_error, w_is_state=True, **optimizer_kwargs)

    # hyperparameters part!
    algorithmic_hyperparameters = []
    eta = tr_dynamics.learning_rate
    if isinstance(eta, tf.Variable):
        if mode != 'reverse':
            algorithmic_hyperparameters.append([eta, tr_dynamics.d_dynamics_d_learning_rate()])
        else:
            algorithmic_hyperparameters.append(eta)
    if hasattr(tr_dynamics, 'momentum_factor'):
        mu = tr_dynamics.momentum_factor
        if isinstance(mu, tf.Variable):
            if mode != 'reverse':
                algorithmic_hyperparameters.append([mu, tr_dynamics.d_dynamics_d_momentum_factor])
            else:
                algorithmic_hyperparameters.append(mu)

    regularization_hyperparameters = []
    vec_w = s.var_list(rf.Vl_Mode.TENSOR)[0]  # vectorized representation of model weights (always the first!)
    if rho_l1s is not None:
        if mode != 'reverse':
            regularization_hyperparameters += [(r1, tr_dynamics.d_dynamics_d_linear_loss_term(
                tf.gradients(er1, vec_w))) for r1, er1 in zip(rho_l1s, reg_l1s)]
        else: regularization_hyperparameters += rho_l1s
    if rho_l2s is not None:
        if mode != 'reverse':
            regularization_hyperparameters += [(r2, tr_dynamics.d_dynamics_d_linear_loss_term(
                tf.gradients(er2, vec_w))) for r2, er2 in zip(rho_l2s, reg_l2s)]
        else: regularization_hyperparameters += rho_l2s
    # end of hyperparameters

    # create hyper_dict
    hyper_dict = {error: regularization_hyperparameters}
    if algo_hyper_wrt_tr_error:  # it is possible to optimize different hyperparameters wrt different validation errors
        hyper_dict[training_error] = algorithmic_hyperparameters
    else: hyper_dict[error].append(algorithmic_hyperparameters)

    hyper_gradients = rf.ReverseHyperGradient(tr_dynamics, hyper_dict) if mode == 'reverse' else \
        rf.ForwardHyperGradient(tr_dynamics, hyper_dict)

    hyper_optimizers = rf.create_hyperparameter_optimizers(hyper_gradients, hyper_optimizer, **hyper_optimizer_kwargs)
    positivity = rf.positivity(hyper_gradients.hyper_list)

    # builds an instance of Real Time Hyperparameter optimization if mode is rtho
    # RealTimeHO exploits partial hypergradients calculated with forward-mode to perform hyperparameter updates
    # while the model is training...
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

    hyper_grads = hyper_gradients.hyper_gradients_dict
    # create a Saver object
    saver = Saver(
        'step', lambda step: step,
        'test accuracy', accuracy, test_supplier,
        'validation accuracy', accuracy,val_supplier,
        'training accuracy', accuracy, tr_supplier,
        'validation error', error, val_supplier,
        *rf.flatten_list([rf.simple_name(hyp), [hyp, hyper_grads[hyp]]]
                      for hyp in hyper_gradients.hyper_list),
        do_print=do_print, collect_data=collect_data
    )

    with tf.Session().as_default() as ss:
        saver.timer.start()
        if __name__ == '__main__':
            if mode == 'rtho':  # here we do not have hyper-iterations
                rtho.initialize()  # helper for initializing all variables...
                for k in range(hyper_iterations):
                    rtho.hyper_batch(hyper_batch_size, train_feed_dict_supplier=tr_supplier, val_feed_dict_suppliers=
                    {error: val_supplier, training_error: all_training_supplier})

                    saver.save(k, append_string=mode)

            else:  # here we do complete hyper-iterations..
                for k in range(hyper_iterations):
                    hyper_gradients.run_all(ev_data.T, train_feed_dict_supplier=tr_supplier, val_feed_dict_suppliers=
                    {error: val_supplier, training_error: all_training_supplier})

                    # update hyperparameters
                    [ss.run(hod.assign_ops) for hod in hyper_optimizers]
                    [ss.run(prj) for prj in positivity]

                    saver.save(k, append_string=mode)


if __name__ == '__main__':
    experiment(None, collect_data=False)