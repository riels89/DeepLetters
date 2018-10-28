import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from model import model
import cv2
import math
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('CNN')
import deepletters_cnn as model

mean_image = np.load('CNN/mean_image.npy')

X = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name='inputs')
y = tf.placeholder(tf.int64, [None], name='labels')
is_training = tf.placeholder(tf.bool, name='is_training')

loss3_SLclassifier_1, loss2_SLclassifier_1, loss1_SLclassifier_1 = model.KitModel(weight_file ='auto_gen/weights.npy',
                                                                                        X=X, is_training=True)
#mean_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), y_out)

real_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), loss3_SLclassifier_1)
aux_loss2 = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), loss2_SLclassifier_1)
aux_loss1 = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), loss1_SLclassifier_1)

total_loss = real_loss + .3 * aux_loss2 + .3 * aux_loss1

optimzer = tf.train.AdamOptimizer(learning_rate=1e-4)

# have tensorflow compute accuracy
correct_prediction = tf.equal(tf.argmax(loss3_SLclassifier_1, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimzer.minimize(total_loss)

def get_pictures(Xd):
    images = np.empty((Xd.shape[0], 227, 227, 3))
    for i in range(Xd.shape[0]):
        im = cv2.imread('data_heap/' + Xd[i])
        images[i] = im
    assert not np.any(np.isnan(images))
    images = images - mean_image
    return images

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):


    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [total_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: get_pictures(Xd[idx]),
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # print(actual_batch_size)
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration " + str(iter_cnt) + ": with minibatch training loss = " + str(loss) + " and accuracy of " + str(np.sum(corr) / actual_batch_size))

            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        epoch_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch " + str(e + 1) + ", Overall loss = " + str(epoch_loss) + " and accuracy of " + str(total_correct))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return epoch_loss, total_correct

def train():

    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
            # tf.train.write_graph(sess.graph_def, 'CNN',
            #          'saved_model.pbtxt', as_text=True)
            clasification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
            inputs = {}
            )
            )

            builder = tf.saved_model.builder.SavedModelBuilder('CNN/saved_model.pb')
            builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            signature_def_map= {
            'predict_images':
            }
            )

            # data = pd.read_csv('227X227.csv')
            #
            # X_data = np.array(data['file_name'])
            # y_data = np.array(data['Letter'])
            #
            # X_train = X_data[:55000]
            #
            # label_encoder = LabelEncoder()
            # y_train = label_encoder.fit_transform(y_data[:55000])
            #
            # X_val = X_data[55000:]
            # y_val = y_data[55000:]
            #
            # sess.run(tf.global_variables_initializer())
            # print('Training')
            # run_model(sess, loss3_SLclassifier_1, total_loss, X_train, y_train, 1, 128, 100, train_step, True)
            # print('Validation')
            # run_model(sess, loss3_SLclassifier_1, total_loss, X_val, y_val, 1, 128)

# data = pd.read_csv('C:/Users/Riley/DeepLettersData/data_heap/128X128.csv')
# X_train = np.array(data['file_name'])[:64]
# y_train = np.array(data['Letter'])
#
# get_pictures(X_train)
train()