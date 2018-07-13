import tensorflow as tf

class model(object):

    def __init__():
        pass

    def create_model(X, is_training):
        X = tf.reshape(X, [-1, 128, 128, 3])
        conv1 = tf.layers.conv2d(inputs=X,
                                 filters=32,
                                 kernal_size=[3, 3],
                                 strides=(1, 1),
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="Convolutional_Layer_1")
        bn1 = tf.layers.batch_normalization(inputs=conv1,
                                            training=is_training,
                                            name="Batch_normalization_1")
        conv2 = tf.layers.conv2d(inputs=bn1,
                                 filters=50,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='Convolutional_layer2')
        bn2 = tf.layers.batch_normalization(inputs=conv2,
                                            training=is_training,
                                            name="Batch_normalization_2")

        max_pool1 = tf.layers.max_pooling2d(bn2,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            padding="valid",
                                            name="Max_pooling_1")

        conv3 = tf.layers.conv2d(inputs=max_pool1,
                                 filters=50,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='Convolutional_layer_3')

        bn3 = tf.layers.batch_normalization(inputs=conv3,
                                            training=is_training,
                                            name="Batch_normalization_3")

        conv4 = tf.layers.conv2d(inputs=bn3,
                                 filters=15,
                                 kernel_size=(1, 1),
                                 strides=(1, 1),
                                 padding="same",
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='Convolutional_layer_4')
        bn4 = tf.layers.batch_normalization(inputs=conv4,
                                            training=is_training,
                                            name="Batch_normalization_4")

        max_pool2 = tf.layers.max_pooling2d(bn4,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            padding="valid",
                                            name="Max_pooling_2")
        reshape = tf.reshape(max_pool2, [-1, 15360])
        fc1 = tf.layers.dense(reshape, 2048, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name='FC_1')
        dropout1 = tf.layers.dropout(inputs=fc1, rate=.2, training=is_training, name='dropout_1')

        fc2 = tf.layers.dense(fc1, 2048, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              name='FC_1')
        dropout2 = tf.layers.dropout(inputs=fc2, rate=.2, training=is_training, name='dropout_1')

        return dropout
