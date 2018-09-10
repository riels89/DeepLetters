import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from model import model
import cv2
import math
from sklearn.preprocessing import LabelEncoder

X = tf.placeholder(tf.float32, [None, 128, 128, 3], name='X')
y = tf.placeholder(tf.int64, [None], name='y')
is_training = tf.placeholder(tf.bool, name='is_training')

y_out = model().create_model(X, is_training)

mean_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), y_out)
optimzer = tf.train.AdamOptimizer(learning_rate=1e-4)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimzer.minimize(mean_loss)

def get_pictures(Xd):
    images = np.empty((Xd.shape[0], 128, 128, 3))
    for i in range(Xd.shape[0]):
        im = cv2.imread('C:/Users/riley/DeepLettersData/data_heap/128x128/' + Xd[i])
        images[i] = im
    return images
def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, correct_prediction, accuracy]
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

            #print(actual_batch_size)
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct

def train():

    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
            data = pd.read_csv('C:/Users/riley/DeepLettersData/data_heap/128X128.csv')

            X_train = np.array(data['file_name'])[:1000]
            #mean_image = np.mean(X_train, axis=0)
            #X_train -= mean_image

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(np.array(data['Letter'][:1000]))

            sess.run(tf.global_variables_initializer())
            print('Training')
            run_model(sess, y_out, mean_loss, X_train, y_train, 10, 64, 100, train_step, True)
            print('Validation')
            #run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)

# data = pd.read_csv('C:/Users/riley/DeepLettersData/data_heap/128X128.csv')
# X_train = np.array(data['file_name'])[:64]
# y_train = np.array(data['Letter'])
#
# get_pictures(X_train)
train()