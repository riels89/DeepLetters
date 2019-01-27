import tensorflow as tf
import logging
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2
import math
from sklearn.preprocessing import LabelEncoder
import sys
import os
import random
import time

sys.path.append('CNN')
import deepletters_cnn as model

log = logging.getLogger()
log.setLevel(logging.DEBUG)

mean_image = np.load('CNN/mean_image.npy')

def get_pictures(Xd, location):
    #print(Xd.shape[0])
    im = cv2.imread(location + "/" + Xd)
    if random.random() < .5:
        im = cv2.flip(im, 0)
        #print(location[i] + "/" + Xd[i])

    assert not np.any(np.isnan(im))
    im = im - mean_image

    return im[np.newaxis]

#X = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name='inputs')
#y = tf.placeholder(tf.int64, [None], name='labels')


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

    with tf.device("/device:GPU:0"):  # "/cpu:0" or "/gpu:0"

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            lr = .001
            relative_root = "CNN/trained_networks/static_v1_lr_test_val-" + str(lr)

            os.mkdir(relative_root)

            fileh = logging.FileHandler(relative_root + "/static_v1_lr-" + str(lr) + ".txt", "a+")
            fileh.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
            fileh.setLevel(logging.DEBUG)

            #loss3_SLclassifier_0 = model.KitModel(weight_file='auto_gen/weights.npy', X=X, mode="lstm")

            saver = tf.train.import_meta_graph("CNN/trained_networks/static_v1_lr-" + str(lr) + "/static_v1_lr-" + str(lr) + ".ckpt.meta")

            #saver = tf.train.Saver()

            saver.restore(sess=sess, save_path="CNN/trained_networks/static_v1_lr-" + str(lr) + "/static_v1_lr-" + str(lr) + ".ckpt")

            graph = tf.get_default_graph()
            loss3_SLclassifier_0 = graph.get_tensor_by_name('pool5/7x7_s1:0')
            loss3_SLclassifier_0 = tf.contrib.layers.flatten(loss3_SLclassifier_0)

            X = graph.get_tensor_by_name('inputs:0')
            y = graph.get_tensor_by_name('labels:0')

            data = pd.read_csv('227X227.csv')

            X_data = np.array(data['file_name'])
            y_data = np.array(data['Letter'])
            location = np.array(data['dir_name'])

            label_encoder = LabelEncoder()

            #65773
            X_val = X_data[0:65773]
            y_val = label_encoder.fit_transform(y_data[0:65773])
            val_loc = location[0:65773]

            print([n.name for n in tf.get_default_graph().as_graph_def().node])

            for folder in ['A', 'B', 'C', 'D', 'E', 'Part1', 'Part2', 'Part3', 'part4', 'Part5']:
                os.mkdir(relative_root + '/' + folder)

            for i in range(X_val.shape[0]):

                feed_dict = {X: get_pictures(X_val[i], val_loc[i]),
                             y: [y_val[i]]}

                pool_layer = sess.run([loss3_SLclassifier_0], feed_dict=feed_dict)
                print(len(pool_layer))
                print(pool_layer[0].shape)

                np.save(str(relative_root) + "/" + str(y_data[i].rsplit('/', 1)[-1]) + '/' + str(X_val[i][:-3]) +'.npy', pool_layer)




