import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datasets.mnist.loader as mnist
from sklearn.preprocessing import OneHotEncoder

class ANN:
    def __init__(self, layers_size):
        self.costs = []
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(layers_size)
        self.store = {}
        self.X = None
        self.Y = None

    def initialize_parameters(self):
        tf.set_random_seed(1)

        for l in range(1, self.L + 1):
            self.parameters["W" + str(l)] = tf.get_variable("W" + str(l),
                                                            shape=[self.layers_size[l], self.layers_size[l - 1]],
                                                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
            self.parameters["b" + str(l)] = tf.get_variable("b" + str(l), shape=[self.layers_size[l], 1],
                                                            initializer=tf.zeros_initializer())

    def forward(self):
        for l in range(1, len(self.layers_size)):

            if l == 1:
                self.store["Z" + str(l)] = tf.add(tf.matmul(self.parameters["W" + str(l)], tf.transpose(self.X)),
                                                  self.parameters["b" + str(l)])
            else:
                self.store["Z" + str(l)] = tf.add(
                    tf.matmul(self.parameters["W" + str(l)], self.store["A" + str(l - 1)]),
                    self.parameters["b" + str(l)])
            if l < self.L:
                self.store["A" + str(l)] = tf.nn.relu(self.store["Z" + str(l)])

        softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(self.store["Z" + str(self.L)]),
                                                             labels=self.Y)

        return softmax

    def fit_predict(self, X_train, Y_train, X_test, Y_test, learning_rate=0.01, n_iterations=2500):
        tf.set_random_seed(1)
        _, f = X_train.shape
        _, c = Y_train.shape

        self.X = tf.placeholder(tf.float32, shape=[None, f], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, c], name='Y')

        self.layers_size.insert(0, f)

        self.initialize_parameters()

        softmax = self.forward()

        cost = tf.reduce_mean(softmax)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_iterations):
                _, epoch_cost = sess.run([optimizer, cost], feed_dict={self.X: X_train, self.Y: Y_train})

                if epoch % 100 == 0:
                    correct_prediction = tf.equal(tf.argmax(self.store["Z" + str(self.L)]),
                                                  tf.argmax(tf.transpose(self.Y)))

                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    print("Cost after epoch %i: %f, Accuracy %f" % (
                        epoch, epoch_cost, accuracy.eval({self.X: X_train, self.Y: Y_train})))

                if epoch % 10 == 0:
                    self.costs.append(epoch_cost)

            correct_prediction = tf.equal(tf.argmax(self.store["Z" + str(self.L)]),
                                          tf.argmax(tf.transpose(self.Y)))

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Test Accuracy %f" % (accuracy.eval({self.X: X_test, self.Y: Y_test})))

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x_orig, train_y_orig, test_x_orig, test_y_orig = mnist.get_data()

    train_x, train_y, test_x, test_y = pre_process_data(train_x_orig, train_y_orig, test_x_orig, test_y_orig)

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    model = ANN(layers_size=[196, 10])
    model.fit_predict(train_x, train_y, test_x, test_y, learning_rate=0.1, n_iterations=1000)
    model.plot_cost()
