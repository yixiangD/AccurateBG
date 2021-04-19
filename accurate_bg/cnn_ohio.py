import numpy as np
import tensorflow as tf
from cgms_data_seg import CGMSDataSeg
from helper import (
    accuracy,
    feature_runner,
    hierarchical_feature_runner,
    hierarchical_runner,
    native_runner,
    runner,
    threeclasses,
)

# from multiprocessing import Pool


def regressor(
    low_fid_data,
    k_size,
    nblock,
    nn_size,
    nn_layer,
    learning_rate,
    batch_size,
    epoch,
    beta,
):

    tf.reset_default_graph()
    sampling_horizon = low_fid_data.sampling_horizon
    feature_size = 0
    if low_fid_data.feature is not None:
        feature_size = low_fid_data.feature[0].size

    x = tf.compat.v1.placeholder(
        tf.float32, [None, sampling_horizon + feature_size], name="x"
    )
    alpha = tf.Variable(tf.random.normal([], stddev=0.1))
    p = tf.math.sin(tf.range(float(sampling_horizon + feature_size)))
    y = x + alpha * p

    assert k_size < sampling_horizon + feature_size
    for _ in range(nblock):
        x0 = tf.slice(y, [0, 0], [-1, 1])
        x0s = tf.tile(x0, [1, k_size - 1])
        xx = tf.concat([x0s, y], 1)
        data = tf.reshape(xx, [-1, sampling_horizon + feature_size + k_size - 1, 1])

        kernel1 = tf.Variable(tf.random.normal([k_size, 1, 1], stddev=0.1))
        kernel2 = tf.Variable(tf.random.normal([k_size, 1, 1], stddev=0.1))
        A = tf.squeeze(tf.nn.conv1d(data, kernel1, 1, "VALID"))
        B = tf.squeeze(tf.nn.conv1d(data, kernel2, 1, "VALID"))
        y = tf.math.multiply(A, tf.sigmoid(B)) + y

    # FNN
    with tf.variable_scope("fnn"):
        W = tf.Variable(
            tf.random.normal([sampling_horizon + feature_size, nn_size], stddev=0.1),
            name="W",
        )
        b = tf.Variable(tf.random.normal([nn_size], stddev=0.1), name="b")
        y = tf.nn.relu(tf.tensordot(y, W, [[1], [0]]) + b)
        for _ in range(nn_layer - 1):
            W = tf.Variable(tf.random.normal([nn_size, nn_size], stddev=0.1), name="W")
            b = tf.Variable(tf.random.normal([nn_size], stddev=0.1), name="b")
            y = tf.nn.relu(tf.tensordot(y, W, [[1], [0]]) + b)

        W = tf.Variable(
            tf.random.normal([nn_size, sampling_horizon], stddev=0.1), name="W"
        )
        b = tf.Variable(tf.random.normal([], stddev=0.1), name="b")
        y = tf.tensordot(y, W, [[1], [0]]) + b

    y = tf.identity(y, name="y")
    y_ = tf.compat.v1.placeholder(tf.float32, [None, sampling_horizon], name="y_")

    weights = tf.compat.v1.placeholder(tf.float32, [sampling_horizon], name="weights")
    # loss = tf.compat.v1.losses.mean_squared_error(
    #    y_, y, weights=tf.expand_dims(weights, axis=0),
    #    reduction=tf.compat.v1.losses.Reduction.MEAN)
    # loss = tf.compat.v1.keras.losses.MeanAbsolutePercentageError()(y_[:, 0], y[:, -1])
    loss = tf.compat.v1.keras.losses.MAE(y_[:, 0], y[:, -1])
    # loss = tf.math.reduce_mean((y_[:, 0] - y[:, -1])**2 / y_[:, 0]**2)

    # add L2 regularization
    L2_var = [
        var
        for var in tf.global_variables()
        if ("fnn/W" in var.name or "fnn/b" in var.name) and "Adam" not in var.name
    ]

    lossL2 = tf.math.add_n([tf.nn.l2_loss(v) for v in L2_var]) * beta

    loss = tf.identity(loss + lossL2, name="loss")

    train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    new_train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=L2_var
    )
    tf.add_to_collections("optimizer", train)
    tf.add_to_collections("optimizer", new_train)

    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(epoch):
        for _ in range(int(low_fid_data.train_n / batch_size)):
            d = low_fid_data.train_next_batch(batch_size)
            sess.run(train, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        print("Epoch %d, train err: %f" % (i, err))
    saver.save(sess, "pretrain")


def test_ckpt(high_fid_data):
    sess = tf.Session()
    saver = tf.train.import_meta_graph("pretrain.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    weights = graph.get_tensor_by_name("weights:0")
    loss = graph.get_tensor_by_name("loss:0")
    y_ = graph.get_tensor_by_name("y_:0")
    y = graph.get_tensor_by_name("y:0")
    d = high_fid_data.test()
    err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
    y_pred = sess.run(y, feed_dict={x: d[0]})
    return err, np.vstack((d[1][:, 0], y_pred[:, -1])).T


def regressor_transfer(high_fid_data, batch_size, epoch, option=1):
    print("------------------in transfer----------------------")
    """
    transfer learning:
    1. reuse seq2seq and FNN weights and train both of them
    2. reuse seq2seq and FNN weights and train FNN weights
    3. reuse seq2seq weights, reinitialize FNN weights and train FNN only
    other: return ErrorMessage
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph("pretrain.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    weights = graph.get_tensor_by_name("weights:0")
    loss = graph.get_tensor_by_name("loss:0")
    y = graph.get_tensor_by_name("y:0")
    y_ = graph.get_tensor_by_name("y_:0")
    # learning_rate = graph.get_tensor_by_name("learn_rate:0")

    if option == 1:
        optimizer = tf.get_collection("optimizer")[0]
    elif option == 2:
        optimizer = tf.get_collection("optimizer")[1]
    elif option == 3:
        optimizer = tf.get_collection("optimizer")[1]
        var = tf.global_variables()
        var_to_init = [
            val
            for val in var
            if ("fnn/W" in val.name or "fnn/b" in val.name) and "Adam" not in val.name
        ]
        epoch *= 3
        sess.run(tf.variables_initializer(var_to_init))
    else:
        print("option not available, please assign 1 or 2 or 3 to option")
        return

    for i in range(epoch):
        for _ in range(int(high_fid_data.train_n / batch_size)):
            d = high_fid_data.train_next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        d = high_fid_data.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        print("Epoch %d, test relative err: %f" % (i, err))
        # print('Epoch %d, test RMSE: %f' % (i, err ** 0.5 / high_fid_data.scale))
    y_pred = sess.run(y, feed_dict={x: d[0]})
    # return err ** 0.5 / high_fid_data.scale, np.reshape(y_pred[:, -1], (-1, 1))
    return err, np.vstack((d[1][:, 0], y_pred[:, -1])).T


def classifier(
    low_fid_data,
    high_fid_data,
    k_size,
    nblock,
    nn_size,
    nn_layer,
    learning_rate,
    batch_size,
    epoch,
    beta,
):

    tf.reset_default_graph()
    batch_size = min(high_fid_data.train_n, batch_size)
    outdim = 3

    learn_rate = tf.constant(learning_rate, name="learn_rate")
    print(f"Learning rate: {learn_rate}")

    sampling_horizon = low_fid_data.sampling_horizon
    if low_fid_data.feature is not None:
        sampling_horizon += low_fid_data.feature[0].size
    sess = tf.compat.v1.Session()

    x = tf.compat.v1.placeholder(tf.float32, [None, sampling_horizon], name="x")
    alpha = tf.Variable(tf.random.normal([], stddev=0.1))
    p = tf.math.sin(tf.range(float(sampling_horizon)))
    y = x + alpha * p
    # weights = tf.compat.v1.placeholder(tf.float32, [sampling_horizon], name="weights")

    assert k_size < sampling_horizon
    for _ in range(nblock):
        x0 = tf.slice(y, [0, 0], [-1, 1])
        x0s = tf.tile(x0, [1, k_size - 1])
        xx = tf.concat([x0s, y], 1)
        data = tf.reshape(xx, [-1, sampling_horizon + k_size - 1, 1])

        kernel1 = tf.Variable(tf.random.normal([k_size, 1, 1], stddev=0.1))
        kernel2 = tf.Variable(tf.random.normal([k_size, 1, 1], stddev=0.1))
        A = tf.squeeze(tf.nn.conv1d(data, kernel1, 1, "VALID"))
        B = tf.squeeze(tf.nn.conv1d(data, kernel2, 1, "VALID"))
        y = tf.math.multiply(A, tf.sigmoid(B)) + y

    # FNN
    with tf.variable_scope("fnn"):
        W = tf.Variable(
            tf.random.normal([sampling_horizon, nn_size], stddev=0.1), name="W"
        )
        b = tf.Variable(tf.random.normal([nn_size], stddev=0.1), name="b")
        y = tf.nn.relu(tf.tensordot(y, W, [[1], [0]]) + b)
        for _ in range(nn_layer - 1):
            W = tf.Variable(tf.random.normal([nn_size, nn_size], stddev=0.1), name="W")
            b = tf.Variable(tf.random.normal([nn_size], stddev=0.1), name="b")
            y = tf.nn.relu(tf.tensordot(y, W, [[1], [0]]) + b)

        W = tf.Variable(tf.random.normal([nn_size, outdim], stddev=0.1), name="W")
        b = tf.Variable(tf.random.normal([], stddev=0.1), name="b")
        y = tf.tensordot(y, W, [[1], [0]]) + b
    y = tf.identity(y, name="y")

    y_ = tf.compat.v1.placeholder(tf.float32, [None, outdim], name="y_")

    loss = tf.losses.softmax_cross_entropy(y_, y)

    # add L2 regularization
    L2_var = [
        var
        for var in tf.global_variables()
        if ("fnn/W" in var.name or "fnn/b" in var.name) and "Adam" not in var.name
    ]

    lossL2 = tf.math.add_n([tf.nn.l2_loss(v) for v in L2_var]) * beta

    loss = tf.identity(loss + lossL2, name="loss")

    train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    new_train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=[W, b]
    )
    tf.add_to_collections("optimizer", train)
    tf.add_to_collections("optimizer", new_train)

    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(epoch):
        for _ in range(int(low_fid_data.train_n / batch_size)):
            d = low_fid_data.train_next_batch(batch_size)
            sess.run(train, feed_dict={x: d[0], y_: threeclasses(d[1][:, None])})
        d = high_fid_data.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: threeclasses(d[1][:, None])})

        y_pred = sess.run(y, feed_dict={x: d[0]})
        acc = accuracy(threeclasses(d[1][:, None]), y_pred)

        print("Epoch %d, %f, test acc: %f" % (i, err, acc))
    saver.save(sess, "pretrain")
    return acc


def classifier_transfer(high_fid_data, batch_size, epoch, option=1):
    print("------------------in transfer----------------------")
    """
    transfer learning:
    1. reuse seq2seq and FNN weights and train both of them
    2. reuse seq2seq and FNN weights and train FNN weights
    3. reuse seq2seq weights, reinitialize FNN weights and train FNN only
    other: return ErrorMessage
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph("pretrain.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    y_ = graph.get_tensor_by_name("y_:0")

    loss = graph.get_tensor_by_name("loss:0")
    train = tf.get_collection("optimizer")[0]
    new_train = tf.get_collection("optimizer")[1]
    if option == 1:
        optimizer = train
    elif option == 2:
        optimizer = new_train
    elif option == 3:
        optimizer = new_train
        var = tf.global_variables()
        var_to_init = [
            val for val in var if ("fnn/W" in val.name or "fnn/b" in val.name)
        ]
        epoch *= 3
        sess.run(tf.variables_initializer(var_to_init))
    else:
        print("option not available, please assign 1 or 2 or 3 to option")
        return

    for i in range(epoch):
        for _ in range(int(high_fid_data.train_n / batch_size)):
            d = high_fid_data.train_next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: d[0], y_: threeclasses(d[1][:, None])})
        d = high_fid_data.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: threeclasses(d[1][:, None])})

        y_pred = sess.run(y, feed_dict={x: d[0]})
        acc = accuracy(threeclasses(d[1][:, None]), y_pred)

        print("Epoch %d, %f, test acc: %f" % (i, err, acc))
    return acc


def main():
    argv = (4, 4, 10, 2, 1e-3, 64, 50, 1e-4)
    series = 100
    native_runner(regressor, argv, regressor_transfer, "Same", series, "CNN")
    # feature_runner(regressor, argv, regressor_transfer, "Same", series, 'CNN')
    # hierarchical_feature_runner(classifier, argv, classifier_transfer, "None", series, 'CNN')


if __name__ == "__main__":
    main()
