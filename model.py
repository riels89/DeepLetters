import tensorflow as tf
import keras

class model(object):

    def __init__():
        pass



    def create_model(features, labels, is_training):
        features = tf.reshape(features, [-1, 128, 128, 3])
        conv1 = tf.layers.conv2d(inputs=features,
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

        max_pool = tf.layers.max_pooling2d(bn2,
                                           pool_size=(2, 2),
                                           strides=(2, 2),
                                           padding="valid",
                                           name="Max_pooling")

        conv3 = tf.layers.conv2d(inputs=max_pool,
                                 filters=50,
                                 kernel_size=(1, 1),
                                 strides=(3, 3),
                                 padding="same",
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='Convolutional_layer_3')

        bn3 = tf.layers.batch_normalization(inputs=conv3,
                                            training=is_training,
                                            name="Batch_normalization_2")

        conv4 = tf.layers.conv2d(inputs=max_pool,
                                 filters=26,
                                 kernel_size=(1, 1),
                                 strides=(2, 2),
                                 padding="same",
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='Convolutional_layer_4')





#
