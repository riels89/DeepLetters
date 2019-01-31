from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2
import math
from sklearn.preprocessing import LabelEncoder
import sys
import os
import random
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging

sys.path.append('CNN')
import deepletters_cnn as model
#tf.logging.set_verbosity(tf.logging.ERROR)




class CNN():
    def __init__(self, absolute_root, learning_rate, resume):
        self.glr = 0
        
        self.learning_rate = learning_rate
                
        self.absolute_root = absolute_root
        
        self.resume = resume
        
        self.mean_image = np.load(self.learning_rate + '/CNN/mean_image.npy')
        
        self.log = logging.getLogger()
        self.log.setLevel(logging.DEBUG)
        


    def create_graph(self, learning_rate=1e-2, mode="testing"):


        X = tf.placeholder(tf.float32, shape=(None, 227, 227, 3), name='inputs')
        y = tf.placeholder(tf.int64, [None], name='labels')

        loss3_SLclassifier_1, loss2_SLclassifier_1, loss1_SLclassifier_1 = model.KitModel(weight_file = self.absolute_root + '/auto_gen/weights.npy',
                                                                                                X=X, mode=mode)
        #mean_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), y_out)

        values, indices = tf.nn.top_k(loss3_SLclassifier_1, 24)

        real_loss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), loss3_SLclassifier_1)
        aux_loss2 = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), loss2_SLclassifier_1)
        aux_loss1 = tf.losses.softmax_cross_entropy(tf.one_hot(y, 24), loss1_SLclassifier_1)

        total_loss = real_loss + .3 * aux_loss2 + .3 * aux_loss1


        optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(loss3_SLclassifier_1, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            # Get only the last layers to train
            FC1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_4c/FC')
            FC2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'incpetion_5a/FC')
            FC3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5b/FC')
            loss1_fc_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'loss1/fc_1')
            loss2_fc_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'loss2/fc_1')
            inception_5a_5x5 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5a/5x5')
            inception_5a_3x3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5a/3x3')
            inception_5b_3x3_reduce = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5b/3x3_reduce')
            inception_5b_5x5_reduce = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5b/5x5_reduce')
            inception_5b_1x1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5b/1x1')
            inception_5b_pool_proj = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5b/pool_proj')
            inception_5b_3x3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5b/3x3')
            inception_5b_5x5 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'inception_5b/5x5')

            train_step = optimzer.minimize(total_loss, var_list=[FC1, FC2, FC3, loss1_fc_1, loss2_fc_1, inception_5a_5x5, inception_5a_3x3, inception_5b_3x3_reduce,
                                           inception_5b_5x5_reduce, inception_5b_1x1, inception_5b_pool_proj, inception_5b_3x3, inception_5b_5x5])
                                           
        return loss3_SLclassifier_1, total_loss, train_step, correct_prediction, accuracy, X, y
    def get_pictures(self, Xd, location):
        images = np.empty((Xd.shape[0], 227, 227, 3))
        #print(Xd.shape[0])
        for i in range(Xd.shape[0]):
            im = cv2.imread(location[i] + "/" + Xd[i])
            if random.random() < .5:
                im = cv2.flip(im, 0)
            #print(location[i] + "/" + Xd[i])
            images[i] = im
        assert not np.any(np.isnan(images))
        images = images - self.mean_image
        return images

    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=64, print_every=100,
                  training=None, plot_losses=False,
                  correct_prediction=None, accuracy=None, X=None, y=None, location=None):


        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [loss_val, correct_prediction, accuracy]
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
                feed_dict = {X: self.get_pictures(Xd[idx], location[idx]),
                             y: yd[idx]}
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
                    self.log.info("Iteration " + str(iter_cnt) + ": with minibatch training loss = " + str(loss) + " and accuracy of " + str(np.sum(corr) / actual_batch_size))

                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            epoch_loss = np.sum(losses) / Xd.shape[0]
            self.log.info("Epoch " + str(e + 1) + ", Overall loss = " + str(epoch_loss) + " and accuracy of " + str(total_correct))
            if plot_losses and e == epochs - 1:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                if training_now is True:
                    plt.savefig(self.absolute_root + "/CNN/trained_networks/static_v1_lr-" + str(glr) +"/static_v1_lr-" + str(glr) +"-TRAIN.png")
                else:
                    plt.savefig(self.absolute_root + "/CNN/trained_networks/static_v1_lr-" + str(glr) +"/static_v1_lr-" + str(glr) + "-VAL.png")
                #plt.show()
        return epoch_loss, total_correct

    def train(self):

        

                # tf.train.write_graph(sess.graph_def, 'CNN',
                #          'saved_model.pbtxt', as_text=True)
                # sess.run(tf.global_variables_initializer())
                #
                # tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
                # tensor_info_y = tf.saved_model.utils.build_tensor_info(loss3_SLclassifier_1)
                #
                # table = tf.contrib.lookup.index_to_string_table_from_tensor(
                #     tf.constant(np.array(list('abcdefghiklmnopqrstuvwxy'))))
                #
                # classification_inputs = tf.saved_model.utils.build_tensor_info(
                #       X)
                # classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
                #       table.lookup(tf.to_int64(indices)))
                # classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)
                #
                # classification_signature = (
                #     tf.saved_model.signature_def_utils.build_signature_def(
                #     inputs={
                #     tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                #           classification_inputs
                #   },
                #   outputs={
                #       tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                #           classification_outputs_classes,
                #       tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                #           classification_outputs_scores
                #   },
                #   method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
                #
                # prediction_signature = (
                #       tf.saved_model.signature_def_utils.build_signature_def(
                #           inputs={'images': tensor_info_x},
                #           outputs={'scores': tensor_info_y},
                #           method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
                #
                #
                # builder = tf.saved_model.builder.SavedModelBuilder('CNN/saved_model_v1.pb')
                # builder.add_meta_graph_and_variables(
                # sess,
                # [tf.saved_model.tag_constants.SERVING],
                # signature_def_map= {
                #     'train':
                #         prediction_signature,
                #     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                #         classification_signature,
                #     },
                # main_op=tf.tables_initializer(),
                # strip_default_attrs=True)
                #
                # builder.save()
        data = pd.read_csv(self.absolute_root + '/227X227.csv')

        X_data = np.array(data['file_name'])
        y_data = np.array(data['Letter'])
        location = np.array(data['dir_name'])

        X_train = X_data[:52995]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_data[:52995])
        train_loc = location[:52995]

        #65773
        X_val = X_data[52995:65773]
        y_val = label_encoder.fit_transform(y_data[52995:65773])
        val_loc = location[52995:65773]

        for lr in self.learning_rate:
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

                with tf.device("/device:GPU:0"):  # "/cpu:0" or "/gpu:0"
                    global glr
                    relative_root = self.absolute_root + "/CNN/trained_networks/static_v1_lr-" + str(lr)

                    glr = lr
                    if self.resume is False:
                        os.mkdir(relative_root)
                    if self.resume and os.path.isdir(relative_root) is False:
                        os.mkdir(relative_root)

                    fileh = logging.FileHandler(relative_root + "/static_v1_lr-" + str(lr) + ".txt", "a+")
                    fileh.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
                    fileh.setLevel(logging.DEBUG)

                    self.log.handlers = [fileh]
                    #log.addHandler(fileh)
                    #log.info('test')

                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

                        loss3_SLclassifier_1, total_loss, train_step, correct_prediction, accuracy, X, y = self.create_graph(lr, mode="training")
                        saver = tf.train.Saver()

                        sess.run(tf.local_variables_initializer())
                        
                        if self.resume and os.path.exists(relative_root + "/static_v1_lr-" + str(lr) + ".ckpt"):
                            saver.restore(sess=sess, save_path=relative_root + "/static_v1_lr-" + str(lr) + ".ckpt")
                        else:
                            sess.run(tf.global_variables_initializer())


                        self.log.info('Training')

                        self.run_model(sess, loss3_SLclassifier_1, total_loss, X_train, y_train, 1, 128, 100, train_step, True, correct_prediction, accuracy, X, y, train_loc)
                        saver.save(sess, relative_root + "/static_v1_lr-" + str(lr) + ".ckpt")

                        self.log.info('Validation')
                        #loss3_SLclassifier_1, total_loss, train_step, correct_prediction, accuracy, X, y = create_graph(lr, False)
                        #scope.reuse_variables()
                        self.run_model(sess, predict=loss3_SLclassifier_1, loss_val=total_loss, Xd=X_val, yd=y_val,
                                  epochs=1, batch_size=128, print_every=100, training=None, plot_losses=True,
                                  correct_prediction=correct_prediction, accuracy=accuracy, X=X, y=y, location=val_loc)



