# import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def get_data():
    return input_data.read_data_sets("../data/minst/", one_hot=True)


def get_placeholders():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')  # 28 * 28
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
    return x, y_


def plot_learning_curves(experiment_data):
    # Generate figure.
    fig, axes = plt.subplots(3, 4, figsize=(22, 12))
    st = fig.suptitle(
        "Learning Curves for all Tasks and Hyper-parameter settings",
        fontsize="x-large")
    # Plot all learning curves.
    for i, results in enumerate(experiment_data):
        for j, (setting, train_accuracy, test_accuracy) in enumerate(results):
            # Plot.
            xs = [x * log_period_samples
                  for x in range(1, len(train_accuracy) + 1)]
            axes[j, i].plot(xs, train_accuracy, label='train_accuracy')
            axes[j, i].plot(xs, test_accuracy, label='test_accuracy')
            # Prettify individual plots.
            axes[j, i].ticklabel_format(
                style='sci', axis='x', scilimits=(0, 0))
            axes[j, i].set_xlabel('Number of samples processed')
            axes[j, i].set_ylabel(
                'Epochs: {}, Learning rate: {}.  Accuracy'.format(*setting))
            axes[j, i].set_title('Task {}'.format(i + 1))
            axes[j, i].legend()
    # Prettify overall figure.
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.91)
    plt.show(False)


log_period_samples = 20000
batch_size = 100
learning_rate = 0.3
num_epochs = 5

# Model1
experiments_tasks1 = []
# (num_epochs, leraning_rate)
settings = [(5, 0.0001), (5, 0.005), (5, 0.1)]

for (num_epochs, learning_rate) in settings:
    tf.reset_default_graph()
    x, y_ = get_placeholders()
    mnist = get_data()
    eval_mnist = get_data()

    initializer = tf.contrib.layers.xavier_initializer()
    w = tf.Variable(initializer([784, 10]))
    b = tf.Variable(initializer([10]))
    logits = tf.matmul(x, w) + b
    y = tf.nn.softmax(logits)
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=logits))
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    i = 0
    train_accuracy = []
    test_accuracy = []
    log_period_updates = int(log_period_samples / batch_size)
    with tf.train.MonitoredSession() as sess:
        while mnist.train.epochs_completed < num_epochs:
            i += 1
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            if i % log_period_updates == 0:
                a = 0.2
                ex = eval_mnist.train.images
                ey = eval_mnist.train.labels
                size = int(ey.shape[0] * a)
                part_ex = ex[0:size, :]
                part_ey = ey[0:size, :]
                train = sess.run(accuracy, feed_dict={x: part_ex, y_: part_ey})
                print("%d th iter train accuracy %f" % (i, train))
                train_accuracy.append(train)
                test = sess.run(
                    accuracy,
                    feed_dict={
                        x: eval_mnist.test.images,
                        y_: eval_mnist.test.labels})
                print("%d th iter test accuracy %f" % (i, test))
                test_accuracy.append(test)
    experiments_tasks1.append(
        ((num_epochs, learning_rate),
         train_accuracy, test_accuracy))

plot_learning_curves([experiments_tasks1])


# Model2

experiments_tasks2 = []
settings = [(15, 0.0001), (15, 0.005), (15, 0.1)]

for (num_epochs, learning_rate) in settings:
    tf.reset_default_graph()
    x, y_ = get_placeholders()
    mnist = get_data()
    eval_mnist = get_data()

    initializer = tf.contrib.layers.xavier_initializer()

    w_1 = tf.Variable(initializer([784, 32]))
    b_1 = tf.Variable(initializer([32]))
    h_1 = tf.nn.relu((tf.matmul(x, w_1) + b_1))

    w_2 = tf.Variable(initializer([32, 10]))
    b_2 = tf.Variable(initializer([10]))
    logits = tf.matmul(h_1, w_2) + b_2
    y = tf.nn.softmax(logits)
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=logits))
    test_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    i, train_accuracy, test_accuracy = 0, [], []
    log_periods_update = int(log_period_samples / batch_size)
    with tf.train.MonitoredSession() as sess:
        while mnist.train.epochs_completed < num_epochs:
            i += 1
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(test_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % log_periods_update == 0:
                a = 0.2
                ex = eval_mnist.train.images
                ey = eval_mnist.train.labels
                size = int(ey.shape[0] * a)
                part_ex = ex[0:size, :]
                part_ey = ey[0:size, :]

                train = sess.run(accuracy, feed_dict={x: part_ex, y_: part_ey})
                print("%d iter train accuracy %f" % (i, train))
                train_accuracy.append(train)

                test = sess.run(
                    accuracy,
                    feed_dict={
                        x: eval_mnist.test.images,
                        y_: eval_mnist.test.labels})
                print("%d iter test accuracy %f" % (i, test))
                test_accuracy.append(test)

        experiments_tasks2.append(
            ((num_epochs, learning_rate),
             train_accuracy, test_accuracy))

# plot_learning_curves([experiments_tasks1, experiments_tasks2])


# Model 3

experiments_tasks3 = []
settings = [(5, 0.003), (40, 0.003), (40, 0.05)]

for num_epochs, learning_rate in settings:
    tf.reset_default_graph()
    x, y_ = get_placeholders()
    mnist = get_data()
    eval_mnist = get_data()

    initializer = tf.contrib.layers.xavier_initializer()
    w_1 = tf.Variable(initializer([784, 32]))
    b_1 = tf.Variable(initializer([32]))
    h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

    w_2 = tf.Variable(initializer([32, 32]))
    b_2 = tf.Variable(initializer([32]))
    h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)

    w_3 = tf.Variable(initializer([32, 10]))
    b_3 = tf.Variable(initializer([10]))
    logits = tf.matmul(h_2, w_3) + b_3
    y = tf.nn.softmax(logits)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    i, train_accuracy, test_accuracy = 0, [], []
    log_period_updates = int(log_period_samples / batch_size)
    with tf.train.MonitoredSession() as sess:
        while mnist.train.epochs_completed < num_epochs:
            i += 1
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _ = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % log_period_updates == 0:
                a = 0.2
                ex = eval_mnist.train.images
                ey = eval_mnist.train.labels
                size = int(ey.shape[0] * a)
                part_ex = ex[0:size, :]
                part_ey = ey[0:size, :]
                train = sess.run(accuracy, feed_dict={x: part_ex, y_: part_ey})
                print("%d iter train accuracy %f" % (i, train))
                train_accuracy.append(train)

                test = sess.run(
                    accuracy,
                    feed_dict={
                        x: eval_mnist.test.images,
                        y_: eval_mnist.test.labels})
                print("%d iter test accuracy %f" % (i, test))
                test_accuracy.append(test)

    experiments_tasks3.append(((num_epochs, learning_rate), train_accuracy, test_accuracy))

plot_learning_curves([experiments_tasks1, experiments_tasks2, experiments_tasks3])

plt.close()
