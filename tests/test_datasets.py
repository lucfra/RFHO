import unittest
from rfho.datasets import *



class RepartitionDatasetTest(unittest.TestCase):

    def test_on_iris(self):

        iris1 = load_iris([.4, .4])
        self.assertEqual(len(iris1.train.data), len(iris1.train.target))
        self.assertEqual(len(iris1.train.data), len(iris1.validation.data))
        self.assertGreater(len(iris1.test.data), 0)

        iris2 = load_iris([.5, .4], classes=2)
        self.assertEqual(len(iris2.test.target[0]), 2)

    def test_on_mnist(self):
        mnist1 = load_mnist(partitions=[.4, .4])
        self.assertEqual(len(mnist1.train.data), len(mnist1.validation.data))
        print(len(mnist1.train.data))

        mnist2 = load_mnist()
        res = redivide_data([mnist2.train, mnist2.validation, mnist2.test])  # do nothing here
        self.assertEqual(len(res[0].data), 55000)
        self.assertEqual(len(res[1].data), 5000)
        self.assertEqual(len(res[2].data), 10000)


class ExampleVisitingTest(unittest.TestCase):

    def test_example_visiting_on_mnist(self):
        mnist = load_mnist()
        ex_v = ExampleVisiting(mnist, batch_size=500, epochs=50)

        ex_v.generate_visiting_scheme()

        self.assertEqual(len(ex_v.training_schedule), ex_v.T*500)

        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        z = tf.placeholder(tf.float32)

        indices = tf.placeholder(tf.int32)

        train_supplier = ex_v.create_train_feed_dict_supplier(x, y, other_feeds={z: 4.},
                                                              lambda_feeds={indices: lambda nb: nb})

        with tf.Session().as_default() as ss:
            print(ss.run(
                [z, indices, x, y], feed_dict=train_supplier(0)
            ))


class LoadDatasetsTest(unittest.TestCase):

    def test_xrmb(self):
        datasets = load_XRMB()

        d0 = datasets.train[0]
        res = []
        for k in range(100):
            num = np.random.randint(0, d0.num_examples)
            speaker = d0.sample_info_dicts[num]['speaker']

            relative_num = reduce(lambda a, v: a - v.num_examples, datasets.train[1:speaker], num)

            res.append(not all(d0.data[num] - datasets.train[speaker].data[relative_num]))
        self.assertTrue(all(res))  # checks that the sample_info_dicts are right


if __name__ == '__main__':
    # ExampleVisitingTest().test_on_mnist()
    unittest.main()
