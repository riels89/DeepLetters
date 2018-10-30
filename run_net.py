import cv2
import time
import tensorflow as tf
import numpy as np
import sys
sys.path.append('auto_gen')
import auto_gen_model as model

def analyze(image):

    X = tf.placeholder(tf.float32, shape = (None, 227, 227, 3), name = 'y')
    prob_1, prob_2, prob_3 = model.KitModel(weight_file='auto_gen/weights.npy', X=X, is_training=True)
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            #image = cv2.imread('index.jpg')
            image = cv2.resize(image, (227, 227))
            image = image[np.newaxis, :, :, :]
            print(image.shape)
            start = time.time()
            sess.run(tf.global_variables_initializer())
            end = time.time()
            print(end - start)
            print(np.argsort(sess.run([prob_1, prob_2, prob_3], feed_dict={X: image})))