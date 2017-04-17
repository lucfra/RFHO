"""
This module contains a simple example of the execution of the three algorithms on various classifiers
trained on mnist
"""
import tensorflow as tf
import rfho as rf


def load_dataset(partition_proportions=(.5,.3)):
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

    return model


def create_logistic_regressor(x, dimensions, **model_kwargs):
    return rf.LinearModel(x, dimensions[0], dimensions[1], **model_kwargs)


def create_ffnn(x, dimensions, **model_kwargs):
    return rf.FFNN(x, dimensions, **model_kwargs)


def prepare_default_models(model, l1=False, l2=False):
    assert isinstance(model, rf.Network)
    res = rf.vectorize_model(model.var_list, model.inp[-1], *model.Ws)
