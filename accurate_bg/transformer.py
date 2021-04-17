import numpy as np
import tensorflow as tf
from helper import (
    accuracy,
    feature_runner,
    hierarchical_runner,
    hypoglycemia,
    native_runner,
    runner,
    threeclasses,
)


def regressor(
    low_fid_data,
    high_fid_data,
    nblock,
    nn_size,
    nn_layer,
    learning_rate,
    batch_size,
    epoch,
    beta,
):

    batch_size = min(high_fid_data.train_n, batch_size)
    tf.reset_default_graph()
    sampling_horizon = low_fid_data.sampling_horizon
    x = tf.compat.v1.placeholder(tf.float32, [None, sampling_horizon], name="x")
    weights = tf.compat.v1.placeholder(tf.float32, [sampling_horizon], name="weights")
    learn_rate = tf.constant(learning_rate, name="learn_rate")
    print(f"Learning rate: {learn_rate}")

    y = x
    for _ in range(nblock):
        y = attention_block(y, sampling_horizon)
        y = tf.squeeze(y)

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

        W = tf.Variable(
            tf.random.normal([nn_size, sampling_horizon], stddev=0.1), name="W"
        )
        b = tf.Variable(tf.random.normal([], stddev=0.1), name="b")
    y = tf.identity(tf.tensordot(y, W, [[1], [0]]) + b, name="y")

    y_ = tf.compat.v1.placeholder(tf.float32, [None, sampling_horizon], name="y_")

    loss = tf.compat.v1.losses.mean_squared_error(
        y_,
        y,
        weights=tf.expand_dims(weights, axis=0),
        reduction=tf.compat.v1.losses.Reduction.MEAN,
    )
    # add L2 regularization
    L2_var = [
        var
        for var in tf.global_variables()
        if ("fnn/W" in var.name or "fnn/b" in var.name) and "Adam" not in var.name
    ]

    lossL2 = tf.math.add_n([tf.nn.l2_loss(v) for v in L2_var]) * beta

    loss = tf.identity(loss + lossL2, name="loss")
    train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.add_to_collections("optimizer", train)

    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(epoch):
        for _ in range(int(low_fid_data.train_n / batch_size)):
            d = low_fid_data.train_next_batch(batch_size)
            sess.run(train, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        d = high_fid_data.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        print("Epoch %d, test RMSE: %f" % (i, err ** 0.5 / low_fid_data.scale))
    saver.save(sess, "pretrain")
    return err ** 0.5 / high_fid_data.scale


def classifier(
    low_fid_data,
    high_fid_data,
    nblock,
    nn_size,
    nn_layer,
    learning_rate,
    batch_size,
    epoch,
    beta,
):

    batch_size = min(high_fid_data.train_n, batch_size)
    outdim = 2
    tf.reset_default_graph()
    sampling_horizon = low_fid_data.sampling_horizon
    if low_fid_data.feature is not None:
        sampling_horizon += low_fid_data.feature[0].size

    x = tf.compat.v1.placeholder(tf.float32, [None, sampling_horizon], name="x")
    # weights = tf.compat.v1.placeholder(tf.float32, [sampling_horizon], name="weights")
    learn_rate = tf.constant(learning_rate, name="learn_rate")
    print(f"Learning rate: {learn_rate}")

    y = x
    for _ in range(nblock):
        y = attention_block(y, sampling_horizon)
        y = tf.squeeze(y)

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

    y = tf.identity(tf.tensordot(y, W, [[1], [0]]) + b, name="y")

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

    tf.add_to_collections("optimizer", train)

    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(epoch):
        low_fid_data.mixup("None")
        for _ in range(int(low_fid_data.train_n / batch_size)):
            d = low_fid_data.train_next_batch(batch_size)
            sess.run(train, feed_dict={x: d[0], y_: hypoglycemia(d[1][:, None])})
        d = high_fid_data.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: hypoglycemia(d[1][:, None])})

        y_pred = sess.run(y, feed_dict={x: d[0]})
        acc = accuracy(hypoglycemia(d[1][:, None]), y_pred)

        print("Epoch %d, %f, test acc: %f" % (i, err, acc))
    saver.save(sess, "pretrain")
    return acc, np.hstack(
        (np.argmax(hypoglycemia(d[1][:, None]), axis=1).reshape((-1, 1)), y_pred)
    )


def classifier_transfer(high_fid_data, batch_size, epoch, option=1):
    print("------------------in transfer----------------------")
    """
    transfer learning:
    1. reuse attention and FNN weights and train both of them
    2. reuse attention and FNN weights and train FNN weights
    3. reuse attention weights, reinitialize FNN weights and train FNN only
    other: return ErrorMessage
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph("pretrain.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    y_ = graph.get_tensor_by_name("y_:0")
    # weights = graph.get_tensor_by_name("weights:0")
    loss = graph.get_tensor_by_name("loss:0")
    learning_rate = graph.get_tensor_by_name("learn_rate:0")

    if option == 1:
        optimizer = tf.get_collection("optimizer")[0]
    elif option > 1:
        var_to_init = [
            var
            for var in tf.global_variables()
            if ("fnn/W" in var.name or "fnn/b" in var.name) and "Adam" not in var.name
        ]
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate, name="newAdam"
        ).minimize(loss, var_list=var_to_init)
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        sess.run(tf.variables_initializer(uninitialized_vars))

        if option == 3:
            # reinitialize weights in FNN
            epoch *= 3
            sess.run(tf.variables_initializer(var_to_init))
    for i in range(epoch):
        for _ in range(int(high_fid_data.train_n / batch_size)):
            d = high_fid_data.train_next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: d[0], y_: hypoglycemia(d[1][:, None])})
        d = high_fid_data.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: hypoglycemia(d[1][:, None])})

        y_pred = sess.run(y, feed_dict={x: d[0]})
        acc = accuracy(hypoglycemia(d[1][:, None]), y_pred)

        print("Epoch %d, %f, test acc: %f" % (i, err, acc))
    return acc, y_pred


def attention_block(x, sampling_horizon):
    # attention
    alpha = tf.Variable(tf.random.normal([], stddev=0.1))
    p = tf.math.sin(tf.range(float(sampling_horizon)))
    xn = x + alpha * p
    xt = tf.reshape(xn, [-1, sampling_horizon, 1])
    xprime = tf.nn.softmax(tf.matmul(xt, tf.transpose(xt, perm=[0, 2, 1])))

    w = tf.Variable(tf.random.normal([], stddev=0.1))
    att = w * tf.matmul(xprime, xt) + xt
    dff = 5
    W = tf.Variable(tf.random.normal([1, dff], stddev=0.1))
    b = tf.Variable(tf.random.normal([dff], stddev=0.1))
    xpprime = tf.nn.relu(tf.tensordot(att, W, [[2], [0]]) + b)
    W = tf.Variable(tf.random.normal([dff, 1], stddev=0.1))
    b = tf.Variable(tf.random.normal([], stddev=0.1))
    y = tf.tensordot(xpprime, W, [[2], [0]]) + b + att
    return y


def regressor_transfer(high_fid_data, batch_size, epoch, option=1):
    print("------------------in transfer----------------------")
    """
    transfer learning:
    1. reuse attention and FNN weights and train both of them
    2. reuse attention and FNN weights and train FNN weights
    3. reuse attention weights, reinitialize FNN weights and train FNN only
    other: return ErrorMessage
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph("pretrain.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./"))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    # y = graph.get_tensor_by_name("y:0")
    y_ = graph.get_tensor_by_name("y_:0")
    weights = graph.get_tensor_by_name("weights:0")
    loss = graph.get_tensor_by_name("loss:0")
    learning_rate = graph.get_tensor_by_name("learn_rate:0")

    if option == 1:
        optimizer = tf.get_collection("optimizer")[0]
    elif option > 1:
        var_to_init = [
            var
            for var in tf.global_variables()
            if ("fnn/W" in var.name or "fnn/b" in var.name) and "Adam" not in var.name
        ]
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate, name="newAdam"
        ).minimize(loss, var_list=var_to_init)
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        sess.run(tf.variables_initializer(uninitialized_vars))

        if option == 3:
            # reinitialize weights in FNN
            epoch *= 3
            sess.run(tf.variables_initializer(var_to_init))
    for i in range(epoch):
        for _ in range(int(high_fid_data.train_n / batch_size)):
            d = high_fid_data.train_next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        d = high_fid_data.test()
        err = sess.run(loss, feed_dict={x: d[0], y_: d[1], weights: d[2]})
        print("Epoch %d, test RMSE: %f" % (i, err ** 0.5 / high_fid_data.scale))
    return err ** 0.5 / high_fid_data.scale


def main():
    arg1 = 8
    arg2 = 10
    arg3 = 2
    argv = (arg1, arg2, arg3, 1e-3, 64, 2, 1e-4)
    series = 1000
    native_runner(classifier, argv, classifier_transfer, "None", series, "Transformer")
    # feature_runner(classifier, argv, classifier_transfer, "None", series, 'Transformer')


if __name__ == "__main__":
    main()
