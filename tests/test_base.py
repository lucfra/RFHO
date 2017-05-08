import tensorflow as tf
from rfho.datasets import load_iris
from rfho.models import vectorize_model, LinearModel
from rfho.utils import cross_entropy_loss




def iris_logistic_regression(augment=0):
    iris = load_iris()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = LinearModel(x, 4, 3)
    model_w, model_y = vectorize_model(model.var_list, model.inp[-1], augment=augment)
    error = tf.reduce_mean(
        cross_entropy_loss(model_y, y)
    )

    correct_prediction = tf.equal(tf.argmax(model_y, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return iris, x, y, model, model_w, model_y, error, accuracy
