# TODO need to be checked after changing ExampleVisiting
"""
This module contains a set of example of the execution of the three main algorithms contained in this package:
- Reverse-HO
- Forward-HO and
- RealTimeHO (RTHO)
 on various classifiers trained on MNIST dataset, with different hyperparameter settings.
"""
import numpy as np
import tensorflow as tf
import rfho as rf
from rfho.datasets import ExampleVisiting


def load_dataset(partition_proportions=(.5, .3)):
    from rfho.datasets import load_mnist
    return load_mnist(partitions=partition_proportions)


IMPLEMENTED_MODEL_TYPES = ['log_reg', 'ffnn', 'cnn']
HO_MODES = ['forward', 'reverse', 'rtho']


def create_model(datasets, model_creator='log_reg', **model_kwargs):
    dataset = datasets.train
    x = tf.placeholder(tf.float32, name='x')
    assert model_creator in IMPLEMENTED_MODEL_TYPES or callable(model_creator), \
        '%s, available %s \n You can implement your own model' \
        'with a function that returns the model! Since a certain structure is assumed ' \
        'in the model object to proceed automatically with the vectorization,' \
        'your model should be a subclass of rf.models.Network...' % (model_creator, IMPLEMENTED_MODEL_TYPES)
    if model_creator == IMPLEMENTED_MODEL_TYPES[0]:
        model = create_logistic_regressor(x, (dataset.dim_data, dataset.dim_target), **model_kwargs)
    elif model_creator == IMPLEMENTED_MODEL_TYPES[1]:  # ffnn deep
        model = create_ffnn(x, dataset.dim_data, dataset.dim_target, **model_kwargs)
    elif model_creator == IMPLEMENTED_MODEL_TYPES[2]:
        model = create_cnn(x, int(np.sqrt(dataset.dim_data)), dataset.dim_target, **model_kwargs)
    else:  # custom _model creator
        model = model_creator(x, **model_kwargs)

    return x, model


def create_logistic_regressor(x, dimensions, **model_kwargs):
    return rf.LinearModel(x, dimensions[0], dimensions[1], **model_kwargs)


def create_ffnn(x, d0, d1, **model_kwargs):
    dimensions = model_kwargs.get('dims', None)
    if dimensions is None: dimensions = [None, 50, 50, 50, 50, None]  # like in MacLaurin (maybe deeper)
    dimensions[0], dimensions[-1] = d0, d1
    model_kwargs['dims'] = dimensions
    return rf.FFNN(x, **model_kwargs)


def create_cnn(x, d0, d1, **model_kwargs):
    model_kwargs.setdefault('conv_dims', [[5, 5, 1, 8], [5, 5, 8, 16]])
    model_kwargs.setdefault('ffnn_dims', [784, 392, d1])

    return rf.SimpleCNN(tf.reshape(x, [-1, d0, d0, 1]), **model_kwargs)


def define_errors_default_models(model, l1=0., l2=0., synthetic_hypers=None, augment=0):
    assert isinstance(model, rf.Network)

    res = rf.vectorize_model(model.var_list, model.inp[-1], *model.Ws,
                             augment=augment)
    s, out, ws = res[0], res[1], res[2:]

    # error
    y = tf.placeholder(tf.float32, name='y')
    error = tf.reduce_mean(rf.cross_entropy_loss(y, out), name='error')  # also validation error

    base_training_error = rf.cross_entropy_loss(y, out)

    gamma = None
    if synthetic_hypers is not None:
        gamma = tf.Variable(tf.ones([synthetic_hypers]))
        training_error = tf.reduce_mean([gamma[k] * base_training_error[k] for k in range(synthetic_hypers)])
    else:
        training_error = tf.reduce_mean(base_training_error)

    rho_l1s, reg_l1s, rho_l2s, reg_l2s = None, None, None, None

    # layer-wise l1 regularizers]
    if isinstance(l1, float):
        rho_l1s = [tf.Variable(l1, name='rho_l1_%d' % k) for k in range(len(ws))]
        reg_l1s = [tf.reduce_sum(tf.abs(w)) for w in ws]
        training_error += tf.reduce_sum([rho * rg_l1 for rho, rg_l1 in zip(rho_l1s, reg_l1s)])

    # layer-wise l2 regularizers]
    if isinstance(l2, float):
        rho_l2s = [tf.Variable(l1, name='rho_l2_%d' % k) for k in range(len(ws))]
        reg_l2s = [tf.reduce_sum(tf.pow(w, 2)) for w in ws]
        training_error += tf.reduce_sum([rho * rg_l1 for rho, rg_l1 in zip(rho_l2s, reg_l2s)])

    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

    return s, out, ws, y, error, training_error, rho_l1s, reg_l1s, rho_l2s, reg_l2s, accuracy, \
           base_training_error, gamma


def experiment_no_saver(datasets=None, model='log_reg', model_kwargs=None, l1=0., l2=0.,
                        synthetic_hypers=None, set_T=None,
                        optimizer=rf.MomentumOptimizer, optimizer_kwargs=None, batch_size=200,
                        algo_hyper_wrt_tr_error=False,
                        mode='reverse', hyper_optimizer=rf.AdamOptimizer, hyper_optimizer_kwargs=None,
                        hyper_iterations=100, hyper_batch_size=100, epochs=None, do_print=True):
    """
    General method for conducting various simple experiments (on MNIST dataset) with RFHO package.

    :param datasets: (some dataset, usually MNIST....)
    :param model: (default logarithmic regression) model type
    :param model_kwargs:
    :param l1: Initial value for l1 regularizer weight (if None does not uses it)
    :param l2: Initial value for l2 regularizer weight (if None does not uses it)
    :param synthetic_hypers: (default None, for benchmarking purposes) if integer adds some synthetic hyperparameters
                                to the task.
    :param set_T:
    :param optimizer: (default `MomentumOptimizer`) optimizer for parameters
    :param optimizer_kwargs:
    :param batch_size:
    :param epochs: number of ephocs
    :param algo_hyper_wrt_tr_error: (default False) if True optimizes the algorithmic hyperparameters (learning rate, ..
                                    w.r.t. training error instead of validation error
    :param mode: forward reverse or rtho
    :param hyper_optimizer: optimizer for the hyperparameters
    :param hyper_optimizer_kwargs:
    :param hyper_iterations: number of hyper-iterations
    :param hyper_batch_size: hyper-batch size when RTHO is used
    :param do_print: if True (default) prints intermediate results...
    :return:
    """
    assert mode in HO_MODES

    if synthetic_hypers:
        batch_size = synthetic_hypers  # altrimenti divento matto!

    if datasets is None: datasets = load_dataset()

    x, model = create_model(datasets, model, **model_kwargs or {})
    s, out, ws, y, error, training_error, rho_l1s, reg_l1s, rho_l2s, reg_l2s, accuracy, base_tr_error, gamma = \
        define_errors_default_models(model, l1, l2, synthetic_hypers=synthetic_hypers,
                                     augment=optimizer.get_augmentation_multiplier())
    tf.add_to_collection('errors', error)

    if optimizer_kwargs is None: optimizer_kwargs = {'lr': tf.Variable(.01, name='eta'),
                                                     'mu': tf.Variable(.5, name='mu')}

    tr_dynamics = optimizer.create(s, loss=training_error, w_is_state=True, **optimizer_kwargs)

    regularization_hyperparameters = []
    if rho_l1s is not None:
        regularization_hyperparameters += rho_l1s
    if rho_l2s is not None:
        regularization_hyperparameters += rho_l2s

    synthetic_hyperparameters = []
    if synthetic_hypers:
        synthetic_hyperparameters.append(gamma)

    hyper_dict = {error: regularization_hyperparameters + synthetic_hyperparameters}  # create hyper_dict
    # end of hyperparameters

    if algo_hyper_wrt_tr_error:  # it is possible to optimize different hyperparameters wrt different validation errors
        hyper_dict[training_error] = tr_dynamics.get_optimization_hyperparameters(only_variables=True)
    else:
        hyper_dict[error] += tr_dynamics.get_optimization_hyperparameters(only_variables=True)

    hyper_opt = rf.HyperOptimizer(tr_dynamics, hyper_dict,
                                  method=rf.ReverseHG if mode == 'reverse' else rf.ForwardHG,
                                  hyper_optimizer_class=hyper_optimizer, **hyper_optimizer_kwargs or {})

    positivity = rf.positivity(hyper_opt.hyper_list)

    # stochastic descent
    ev_data = ExampleVisiting(datasets.train, batch_size=batch_size, epochs=epochs)
    if epochs: ev_data.generate_visiting_scheme()
    tr_supplier = ev_data.create_feed_dict_supplier(x, y)
    val_supplier = datasets.validation.create_supplier(x, y)

    def _all_training_supplier():
        return {x: datasets.train.data, y: datasets.train.target}

    # feed_dict supplier for validation errors
    val_feed_dict_suppliers = {error: val_supplier}
    if algo_hyper_wrt_tr_error: val_feed_dict_suppliers[training_error] = _all_training_supplier

    # number of iterations
    T = set_T or ev_data.T if mode != 'rtho' else hyper_batch_size

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():
        hyper_opt.initialize()
        for k in range(hyper_iterations):
            hyper_opt.run(T, train_feed_dict_supplier=tr_supplier, val_feed_dict_suppliers=val_feed_dict_suppliers,
                          hyper_constraints_ops=positivity)

            if mode != 'rtho':
                hyper_opt.initialize()


def experiment(name_of_experiment, collect_data=False,
               datasets=None, model='log_reg', model_kwargs=None, l1=0., l2=0.,
               synthetic_hypers=None, set_T=None,
               optimizer=rf.MomentumOptimizer, optimizer_kwargs=None, batch_size=200,
               algo_hyper_wrt_tr_error=False,
               mode='reverse', hyper_optimizer=rf.AdamOptimizer, hyper_optimizer_kwargs=None,
               hyper_iterations=100, hyper_batch_size=100, epochs=None, do_print=True):
    """
    General method for conducting various simple experiments (on MNIST dataset) with RFHO package.

    :param name_of_experiment: a name for the experiment. Will be used as root folder for the saver (this is the only
                                positional parameter..)
    :param collect_data: (default False) wheter to save data
    :param datasets: (some dataset, usually MNIST....)
    :param model: (default logarithmic regression) model type
    :param model_kwargs:
    :param l1: Initial value for l1 regularizer weight (if None does not uses it)
    :param l2: Initial value for l2 regularizer weight (if None does not uses it)
    :param synthetic_hypers: (default None, for benchmarking purposes) if integer adds some synthetic hyperparameters
                                to the task.
    :param set_T:
    :param optimizer: (default `MomentumOptimizer`) optimizer for parameters
    :param optimizer_kwargs:
    :param batch_size:
    :param epochs: number of ephocs
    :param algo_hyper_wrt_tr_error: (default False) if True optimizes the algorithmic hyperparameters (learning rate, ..
                                    w.r.t. training error instead of validation error
    :param mode: forward reverse or rtho
    :param hyper_optimizer: optimizer for the hyperparameters
    :param hyper_optimizer_kwargs:
    :param hyper_iterations: number of hyper-iterations
    :param hyper_batch_size: hyper-batch size when RTHO is used
    :param do_print: if True (default) prints intermediate results...
    :return:
    """
    assert mode in HO_MODES

    if synthetic_hypers:
        batch_size = synthetic_hypers  # altrimenti divento matto!

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
        algorithmic_hyperparameters.append(eta)
    if hasattr(tr_dynamics, 'momentum_factor'):
        mu = tr_dynamics.momentum_factor
        if isinstance(mu, tf.Variable):
            algorithmic_hyperparameters.append(mu)

    regularization_hyperparameters = []
    vec_w = s.var_list(rf.VlMode.TENSOR)[0]  # vectorized representation of _model weights (always the first!)
    if rho_l1s is not None:
        regularization_hyperparameters += rho_l1s
    if rho_l2s is not None:
        regularization_hyperparameters += rho_l2s

    synthetic_hyperparameters = []
    if synthetic_hypers:
        synthetic_hyperparameters.append(gamma)

    hyper_dict = {error: regularization_hyperparameters + synthetic_hyperparameters}  # create hyper_dict
    # end of hyperparameters

    if algo_hyper_wrt_tr_error:  # it is possible to optimize different hyperparameters wrt different validation errors
        hyper_dict[training_error] = algorithmic_hyperparameters
    else:
        hyper_dict[error] += algorithmic_hyperparameters
    # print(hyper_dict)

    hyper_opt = rf.HyperOptimizer(tr_dynamics, hyper_dict,
                                  method=rf.ReverseHG if mode == 'reverse' else rf.ForwardHG,
                                  hyper_optimizer_class=hyper_optimizer, **hyper_optimizer_kwargs or {})

    positivity = rf.positivity(hyper_opt.hyper_list)

    # stochastic descent
    ev_data = ExampleVisiting(datasets.train, batch_size=batch_size, epochs=epochs)
    if epochs: ev_data.generate_visiting_scheme()
    tr_supplier = ev_data.create_feed_dict_supplier(x, y)
    val_supplier = datasets.validation.create_supplier(x, y)
    test_supplier = datasets.test.create_supplier(x, y)

    def _all_training_supplier():
        return {x: datasets.train.data, y: datasets.train.target}

    # feed_dict supplier for validation errors
    val_feed_dict_suppliers = {error: val_supplier}
    if algo_hyper_wrt_tr_error: val_feed_dict_suppliers[training_error] = _all_training_supplier

    def _calculate_memory_usage():
        memory_usage = rf.simple_size_of_with_pickle([
            hyper_opt.hyper_gradients.w.eval(),
            [h.eval() for h in hyper_opt.hyper_gradients.hyper_list]
        ])
        if mode == 'reverse':
            return memory_usage + rf.simple_size_of_with_pickle([
                hyper_opt.hyper_gradients.w_hist,
                [p.eval() for p in hyper_opt.hyper_gradients.p_dict.values()]
            ])
        else:
            return memory_usage + rf.simple_size_of_with_pickle([
                [z.eval() for z in hyper_opt.hyper_gradients.zs]
            ])

    # number of iterations
    T = set_T or ev_data.T if mode != 'rtho' else hyper_batch_size

    hyper_grads = hyper_opt.hyper_gradients.hyper_gradients_dict
    # create a Saver object
    if name_of_experiment:
        saver = rf.Saver(name_of_experiment,
                          'step', lambda step: step,
                          'mode', lambda step: mode,
                          'test accuracy', accuracy, test_supplier,
                          'validation accuracy', accuracy, val_supplier,
                          'training accuracy', accuracy, tr_supplier,
                          'validation error', error, val_supplier,
                          'memory usage (mb)', lambda step: _calculate_memory_usage() * 9.5367e-7,
                          'weights', vec_w,
                          '# weights', lambda step: vec_w.get_shape().as_list()[0],
                          '# hyperparameters', lambda step: len(hyper_opt.hyper_list),
                          '# iterations', lambda step: T,
                         *rf.flatten_list([rf.simple_name(hyp), [hyp, hyper_grads[hyp]]]
                                          for hyp in hyper_opt.hyper_list),
                         do_print=do_print, collect_data=collect_data
                         )
    else:
        saver = None

    save_dict_history = []

    with tf.Session(config=rf.CONFIG_GPU_GROWTH).as_default():
        if saver: saver.timer.start()
        hyper_opt.initialize()
        for k in range(hyper_iterations):
            hyper_opt.run(T, train_feed_dict_supplier=tr_supplier, val_feed_dict_suppliers=val_feed_dict_suppliers,
                          hyper_constraints_ops=positivity)

            if saver: save_dict_history.append(saver.save(k, append_string='_%s' % mode))

            if mode != 'rtho':
                hyper_opt.initialize()

    return save_dict_history


def _check_adam():
    for _mode in HO_MODES[:2]:
        for _model in IMPLEMENTED_MODEL_TYPES[1:2]:
            _model_kwargs = {'dims': [None, 300, 300, None]}
            tf.reset_default_graph()

            # set random seeds!!!!
            np.random.seed(1)
            tf.set_random_seed(1)

            experiment('test_with_model_' + _model,
                       collect_data=False, hyper_iterations=3, mode=_mode, epochs=3,
                       optimizer=rf.AdamOptimizer,
                       optimizer_kwargs={'lr': tf.Variable(.001, name='eta_adam')},
                       model=_model,
                       model_kwargs=_model_kwargs,
                       set_T=100,
                       )


def _check_forward():
    w_100 = []
    for i in range(1):
        for _mode in HO_MODES[0:1]:
            for _model in IMPLEMENTED_MODEL_TYPES[0:2]:
                _model_kwargs = {}  # {'dims': [None, 300, 300, None]}
                tf.reset_default_graph()
                # set random seeds!!!!
                np.random.seed(1)
                tf.set_random_seed(1)

                results = experiment('test_with_model_' + _model, collect_data=False, hyper_iterations=10, mode=_mode,
                                     epochs=None,
                                     model=_model,
                                     model_kwargs=_model_kwargs,
                                     set_T=1000,
                                     synthetic_hypers=None,
                                     hyper_batch_size=100
                                     # optimizer=rf.GradientDescentOptimizer,
                                     # optimizer_kwargs={'lr': tf.Variable(.01, name='eta')}
                                     )
                w_100.append(results[0]['weights'])
    # rf.save_obj(w_100, 'check_forward')
    return w_100


def _check_all_methods():
    for _mode in HO_MODES[1:]:
        for _model in IMPLEMENTED_MODEL_TYPES:
            # _model_kwargs = {'dims': [None, 300, 300, None]}
            tf.reset_default_graph()
            # set random seeds!!!!
            np.random.seed(1)
            tf.set_random_seed(1)

            experiment('test_with_model_' + _model, collect_data=False, hyper_iterations=3, mode=_mode,
                       # epochs=3,
                       model=_model,
                       # model_kwargs=_model_kwargs,
                       set_T=100,
                       synthetic_hypers=None,
                       hyper_batch_size=100
                       # optimizer=rf.GradientDescentOptimizer,
                       # optimizer_kwargs={'lr': tf.Variable(.01, name='eta')}
                       )


def _check_cnn():
    print('END')
    for _mode in HO_MODES[2:3]:
        for _model in IMPLEMENTED_MODEL_TYPES[2:3]:
            tf.reset_default_graph()
            np.random.seed(1)
            tf.set_random_seed(1)

            _model_kwargs = {'conv_dims': [[5, 5, 1, 2], [5, 5, 2, 4], [5, 5, 4, 8]],
                             'ffnn_dims': [128, 10]}

            # noinspection PyTypeChecker
            experiment('test_with_model_' + _model, collect_data=False, hyper_iterations=3, mode=_mode,
                       epochs=2,
                       model=_model,
                       model_kwargs=_model_kwargs,
                       set_T=100,
                       synthetic_hypers=None,
                       hyper_batch_size=100,
                       l1=None,
                       l2=None
                       # optimizer=rf.GradientDescentOptimizer,
                       # optimizer_kwargs={'lr': tf.Variable(.01, name='eta')}
                       )


def _check_new_saver_mode():
    saver = rf.Saver('TBD2')
    datasets = load_dataset()

    with rf.Records.on_hyperiteration(saver, rf.Records.hyperparameters(), rf.Records.hypergradients(),
                                  rf.Records.tensors('error', 'accuracy', rec_name='valid',
                                                     fd=('x', 'y', datasets.validation)),
                                  rf.Records.tensors('error', 'accuracy', rec_name='test',
                                                     fd=('x', 'y', datasets.test))
                                  ):
        with rf.Records.on_forward(saver, rf.Records.norms_of_z(), rf.Records.norms_of_d_dynamics_d_hypers(),
                                   append_string='zs', do_print=False):
            experiment_no_saver(datasets=datasets, mode=HO_MODES[0],
                                epochs=None, set_T=100, hyper_iterations=5)

    experiment_no_saver(mode=HO_MODES[1], epochs=None, set_T=10, hyper_iterations=3)


if __name__ == '__main__':
    _check_all_methods()
    # _check_forward()
    #  _check_adam()
    # [_check_cnn() for _ in range(3)]
    _check_new_saver_mode()
