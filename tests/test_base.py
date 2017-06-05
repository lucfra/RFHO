import tensorflow as tf
from rfho.datasets import load_iris
import numpy as np
import rfho as rf

TRACK = 'TRACK'


def iris_logistic_regression(augment=0):
    """
    Simple model for testing purposes
    
    :param augment: 
    :return: 
    """
    iris = load_iris(partitions_proportions=(.3,.3))
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = rf.LinearModel(x, 4, 3)
    model_w, model_y = rf.vectorize_model(model.var_list, model.inp[-1], augment=augment)
    error = tf.reduce_mean(
        rf.cross_entropy_loss(model_y, y)
    )

    correct_prediction = tf.equal(tf.argmax(model_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return iris, x, y, model, model_w, model_y, error, accuracy


def track_tensors(*tensors):
    # print(tensors)
    with tf.name_scope(TRACK):
        [tf.identity(t, name=rf.simple_name(t)) for t in tensors]


def assert_array_list_all_same(lst, raise_error=True, msg=''):
    e0 = lst[0]
    for k, e in enumerate(lst[1:]):
        if raise_error:
            assert np.array_equal(e0, e), msg + 'difference found at %d' % k
        else:
            if not np.array_equal(e0, e):
                print(msg + 'difference found at %d' % k)
                return k


def assert_array_lists_same(lst1, lst2, raise_error=True, msg='', print_differences=True, test_case=None):
    for (k, (e1, e2)) in enumerate(zip(lst1, lst2)):
        if raise_error:
            if test_case:
                test_case.assertTrue(np.array_equal(e1, e2))
            else:
                assert np.array_equal(e1, e2), msg + 'difference found at %d' % k
        else:
            if not np.array_equal(e1, e2):
                print(msg + 'difference found at %d' % k)
                if print_differences:
                    print(e1 - e2)
                return k
    return None
