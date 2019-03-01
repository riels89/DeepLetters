import tensorflow as tf
import logging
import numpy as np
import cv2
import sys
import os
import random
import pandas as pd

sys.path.append('CNN')

log = logging.getLogger()
log.setLevel(logging.DEBUG)

mean_image = np.load('CNN/mean_image.npy')

def get_pictures(location):

    cap = cv2.VideoCapture(location)
    video = np.empty([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 227, 227, 3])
    fliped = random.random() < .5
    for i in range(video.shape[0]):
        ret, frame = cap.read()
        resized_image = cv2.resize(frame, (227, 227))
        resized_image = resized_image - mean_image

        if fliped:
            resized_image = cv2.flip(resized_image, 0)

        video[i] = resized_image

    cap.release()
    cv2.destroyAllWindows()
    assert not np.any(np.isnan(video))

    return video

#X = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name='inputs')
#y = tf.placeholder(tf.int64, [None], name='labels')


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

    with tf.device("/device:GPU:0"):  # "/cpu:0" or "/gpu:0"

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            lr = 1e-6
            relative_root = "CNN/trained_networks/static_v2_lr_test_val-" + str(lr)

            saver = tf.train.import_meta_graph("CNN/trained_networks/static_v2_lr-" + str(lr) + "/epoch-20/static_v2_lr-" + str(lr) + ".ckpt.meta")

            #saver = tf.train.Saver()

            saver.restore(sess=sess, save_path="CNN/trained_networks/static_v2_lr-" + str(lr) + "/epoch-20/static_v2_lr-" + str(lr) + ".ckpt")

            graph = tf.get_default_graph()
            loss3_SLclassifier_0 = graph.get_tensor_by_name('pool5/7x7_s1:0')
            loss3_SLclassifier_0 = tf.contrib.layers.flatten(loss3_SLclassifier_0)

            X = graph.get_tensor_by_name('inputs:0')
            y = graph.get_tensor_by_name('labels:0')

            #print([n.name for n in tf.get_default_graph().as_graph_def().node])

            csv = pd.DataFrame(columns=['word', 'filepath', 'signer'])

            for signer in ['Single']:
                for root, dirnames, filenames in os.walk("video_data/" + str(signer)):
                    for filename in filenames:

                        full_path = os.path.join(root, filename)

                        feed_dict = {X: get_pictures(full_path)}

                        csv = csv.append({'word': filename[:-4], 'filepath': "video_data/numpy/" + str(filename[:-4]) + '.npy',
                                    'signer': signer}, ignore_index=True)

                        pool_layers = sess.run([loss3_SLclassifier_0], feed_dict=feed_dict)

                        print(csv)

                        np.save("video_data/numpy/" + str(filename[:-4]) + '.npy', pool_layers)

            csv.to_csv('pool_layers.csv')
