import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


class Solver(object):

    def __init__():
        total_loss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=y_out)
        mean_loss = tf.reduce_mean(total_loss)

        # define our optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)  # select optimizer and set learning rate
        train_step = optimizer.minimize(mean_loss)

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
                feed_dict = {X: Xd[idx, :],
                             y: yd[idx],
                             is_training: training_now}
                # get batch size
                actual_batch_size = yd[idx].shape[0]

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
