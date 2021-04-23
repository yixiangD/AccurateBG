import os

import numpy as np
import tensorflow as tf
from cgms_data_seg import CGMSDataSeg

# from multiprocessing import Pool


def regressor(
    low_fid_data,
    k_size,
    nblock,
    nn_size,
    nn_layer,
    start_learning_rate,
    batch_size,
    epoch,
    beta,
    loss_type,
    outdir,
):

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
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
        A = tf.squeeze(
            tf.nn.conv1d(input=data, filters=kernel1, stride=1, padding="VALID")
        )
        B = tf.squeeze(
            tf.nn.conv1d(input=data, filters=kernel2, stride=1, padding="VALID")
        )
        y = tf.math.multiply(A, tf.sigmoid(B)) + y

    # FNN
    with tf.compat.v1.variable_scope("fnn"):
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
    assert loss_type in ["mse", "mape", "mae", "relative_mse", "rmse"]
    if loss_type == "mse":
        loss = tf.compat.v1.losses.mean_squared_error(
            y_,
            y,
            weights=tf.expand_dims(weights, axis=0),
            reduction=tf.compat.v1.losses.Reduction.MEAN,
        )
    elif loss_type == "mape":
        loss = tf.compat.v1.keras.losses.MeanAbsolutePercentageError()(
            y_[:, 0], y[:, -1]
        )
    elif loss_type == "mae":
        loss = tf.compat.v1.keras.losses.MAE(y_[:, 0], y[:, -1])
    elif loss_type == "relative_mse":
        loss = tf.math.reduce_mean(
            input_tensor=(y_[:, 0] - y[:, -1]) ** 2 / y_[:, 0] ** 2
        )
    elif loss_type == "rmse":
        loss = tf.math.sqrt(
            tf.math.reduce_mean(input_tensor=(y_[:, 0] - y[:, -1]) ** 2)
        )

    # add L2 regularization
    L2_var = [
        var
        for var in tf.compat.v1.global_variables()
        if ("fnn/W" in var.name or "fnn/b" in var.name) and "Adam" not in var.name
    ]

    lossL2 = tf.math.add_n([tf.nn.l2_loss(v) for v in L2_var]) * beta

    loss = tf.identity(loss + lossL2, name="loss")
    learning_rate = tf.compat.v1.train.exponential_decay(
        start_learning_rate, epoch, epoch / 5, 0.96
    )

    train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
    new_train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=L2_var
    )
    tf.compat.v1.add_to_collections("optimizer", train)
    tf.compat.v1.add_to_collections("optimizer", new_train)

    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(epoch):
        for _ in range(int(low_fid_data.train_n / batch_size)):
            d = low_fid_data.train_next_batch(batch_size)
            sess.run(train, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        print("Epoch %d, train loss: %f" % (i, err))
    saver.save(sess, os.path.join(outdir, "pretrain"))


def test_ckpt(high_fid_data, outdir):
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(outdir, "pretrain.meta"))
    saver.restore(sess, tf.train.latest_checkpoint(outdir))

    graph = tf.compat.v1.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    weights = graph.get_tensor_by_name("weights:0")
    loss = graph.get_tensor_by_name("loss:0")
    y_ = graph.get_tensor_by_name("y_:0")
    y = graph.get_tensor_by_name("y:0")
    d = high_fid_data.test()
    err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
    y_pred = sess.run(y, feed_dict={x: d[0]})
    return err, np.vstack((d[1][:, 0], y_pred[:, -1])).T


def regressor_transfer(
    train_dataset, test_dataset, batch_size, epoch, outdir, option=1
):
    print("------------------in transfer----------------------")
    """
    transfer learning:
    1. reuse seq2seq and FNN weights and train both of them
    2. reuse seq2seq and FNN weights and train FNN weights
    3. reuse seq2seq weights, reinitialize FNN weights and train FNN only
    other: return ErrorMessage
    """
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(outdir, "pretrain.meta"))
    saver.restore(sess, tf.train.latest_checkpoint(outdir))

    graph = tf.compat.v1.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    weights = graph.get_tensor_by_name("weights:0")
    loss = graph.get_tensor_by_name("loss:0")
    y = graph.get_tensor_by_name("y:0")
    y_ = graph.get_tensor_by_name("y_:0")

    if option == 1:
        optimizer = tf.compat.v1.get_collection("optimizer")[0]
    elif option == 2:
        optimizer = tf.compat.v1.get_collection("optimizer")[1]
    elif option == 3:
        optimizer = tf.compat.v1.get_collection("optimizer")[1]
        var = tf.compat.v1.global_variables()
        var_to_init = [
            val
            for val in var
            if ("fnn/W" in val.name or "fnn/b" in val.name) and "Adam" not in val.name
        ]
        epoch *= 3
        sess.run(tf.compat.v1.variables_initializer(var_to_init))
    else:
        print("option not available, please assign 1 or 2 or 3 to option")
        return

    for i in range(epoch):
        for _ in range(int(train_dataset.train_n / batch_size)):
            d = train_dataset.train_next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        d = test_dataset.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        print("Epoch %d, test loss: %f" % (i, err))
    y_pred = sess.run(y, feed_dict={x: d[0]})
    return err, np.vstack((d[1][:, 0], y_pred[:, -1])).T
