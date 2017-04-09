import tensorflow as tf
from rfho.utils import hvp
from rfho.models import LinearModel, vectorize_model
from rfho.datasets import load_iris
import numpy as np



def test_hv_with_builtin():
    iris = load_iris()
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')
    model = LinearModel(x, 4, 3)
    net_w, net_out = vectorize_model(model.var_list, model.inp[-1])

    v = tf.constant(np.ones(net_w.tensor.get_shape()), dtype=tf.float32)  # vector of ones of right shape

    ce_builtin = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=net_out, labels=y)
    )  # this is the builtin function advertised on tensorflow for computing cross entropy loss with softmax output

    ce_standard = tf.reduce_mean(
        -tf.reduce_sum(
            y*tf.log(tf.nn.softmax(net_out)), reduction_indices=[1]
        )  # this is normal CE loss
    )

    hvp_builtin = hvp(ce_builtin, net_w.tensor, v)  # WITH PREVIOUS VERSIONS (r.0.11) WAS 0. NOW RAISES ERROR
    hessian_builtin = tf.hessians(ce_builtin, net_w)[0]

    hvp_standard = hvp(ce_standard, net_w, v)
    hessian_standard = tf.hessians(ce_standard, net_w)[0]

    def training_supplier(): return {x: iris.train.data, y: iris.train.target}

    ts = tf.train.GradientDescentOptimizer(.1).minimize(ce_standard, var_list=[net_w])

    with tf.Session().as_default() as ss:
        tf.global_variables_initializer().run()

        print('builtin, standard:', ss.run([ce_builtin, ce_standard], feed_dict=training_supplier()))

        for _ in range(2000):
            ts.run(feed_dict=training_supplier())

        print('builtin', ss.run([hvp_builtin, hessian_builtin], feed_dict=training_supplier()))  # output is wrongly 0.

        print('standard', ss.run([hvp_standard, hessian_standard], feed_dict=training_supplier()))


if __name__ == '__main__':
    test_hv_with_builtin()
