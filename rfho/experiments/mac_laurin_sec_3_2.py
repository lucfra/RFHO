import tensorflow as tf
import numpy as np
from rfho.datasets import load_mnist, Datasets, Dataset
from rfho.utils import cross_entropy_loss

from rfho.models import LinearModel, vectorize_model, ffnn_lin_out, FFNN, ffnn_layer


def build_model(augment=0, variable_initializer=(tf.zeros, tf.zeros)):
    mnist = load_mnist()
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    lin_model = LinearModel(x, 28 * 28, 10,
                            active_gen=ffnn_lin_out(variable_initializer[0], variable_initializer[1]))

    all_w, mod_y, mat_w = vectorize_model(lin_model.var_list, lin_model.inp[-1], lin_model.Ws[0], augment=augment)

    error = tf.reduce_mean(
        cross_entropy_loss(mod_y, y)
    )

    correct_prediction = tf.equal(tf.argmax(mod_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return mnist, x, y, all_w, mod_y, mat_w, error, accuracy


def exp_multitask_primary_secondary_function_build_models(train, valid, test, n1=5, n_int=10, n2_1=1, n2_2=3,
                                                          hu1=10, hu2_1=5, hu2_2=3):
    x = tf.placeholder(tf.float32)
    h_layer_gen = ffnn_layer(init_w=lambda shape: tf.random_normal(shape, stddev=5.),
                             init_b=tf.random_normal, activ=tf.tanh)
    linear_layer = ffnn_layer(init_w=tf.random_normal, init_b=tf.zeros, activ=tf.identity)
    phi = FFNN(x, [n1, hu1, n_int], activ_gen=(h_layer_gen, linear_layer))  # shared representation

    zero_one_layer = ffnn_layer(init_w=tf.random_normal, init_b=tf.zeros,
                                activ=lambda features, name: (1. + tf.sign(
                                    tf.transpose(
                                        tf.transpose(features) - tf.reduce_max(features, reduction_indices=[1])
                                    ) + 1.e-10, name))/2)

    f = FFNN(phi.inp[-1], [n_int, hu2_2, n2_2], activ_gen=(h_layer_gen, zero_one_layer))  # primary task

    g = FFNN(phi.inp[-1], [n_int, hu2_1, n2_1], activ_gen=(linear_layer, linear_layer))  # secondary task

    def rnd_func(exn): return np.random.uniform(-1., 1., size=(exn, n1))  # generate points
    train_x, valid_x, test_x = (rnd_func(train), rnd_func(valid), rnd_func(test))

    with tf.Session().as_default() as ss:
        tf.global_variables_initializer().run()
        train_f_y, train_g_y = ss.run([f.inp[-1], g.inp[-1]], feed_dict={x: train_x})
        valid_f_y, valid_g_y = ss.run([f.inp[-1], g.inp[-1]], feed_dict={x: valid_x})
        test_f_y, test_g_y = ss.run([f.inp[-1], g.inp[-1]], feed_dict={x: test_x})

    train_dataset_f = Dataset(data=train_x, target=train_f_y)
    valid_dataset_f = Dataset(data=valid_x, target=valid_f_y)
    test_dataset_f = Dataset(data=test_x, target=test_f_y)
    dataset_f = Datasets(train=train_dataset_f, validation=valid_dataset_f, test=test_dataset_f)

    train_dataset_g = Dataset(data=train_x, target=train_g_y)
    valid_dataset_g = Dataset(data=valid_x, target=valid_g_y)
    test_dataset_g = Dataset(data=test_x, target=test_g_y)
    dataset_g = Datasets(train=train_dataset_g, validation=valid_dataset_g, test=test_dataset_g)

    return dataset_f, dataset_g


if __name__ == '__main__':
    data_f, data_g = exp_multitask_primary_secondary_function_build_models(10, 2, 2)

    print(data_f.train.data)
    print(data_f.train.target)

