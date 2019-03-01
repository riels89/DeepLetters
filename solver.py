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
#import seaborn as sns
from tensorflow.python.tools import inspect_checkpoint as chkp


sys.path.append('CNN')
import deepletters_cnn as model
#tf.logging.set_verbosity(tf.logging.ERROR)




class CNN():
    def __init__(self, absolute_root, learning_rate, resume, momentum, batch_size, start_epoch):
        self.glr = 0

        self.momentum = momentum

        self.learning_rate = learning_rate

        self.absolute_root = absolute_root

        self.resume = resume
        #/model/deephand_mean.npy
        self.mean_image = np.resize(np.load(absolute_root + '/CNN/mean_image.npy'), [227,227,3])

        self.log = logging.getLogger()
        self.log.setLevel(logging.DEBUG)

        self.batch_size = batch_size

        self.start_epoch = start_epoch

        #self.secondary_lr = secondary_lr

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

        vars = tf.trainable_variables()

        # lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
        #               if 'bias' not in v.name ]) * 0.01
        #
        # total_loss = total_loss + lossL2

        #optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimzer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum, use_nesterov=True)
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(loss3_SLclassifier_1, 1), y)
        top_5_corr = tf.nn.in_top_k(predictions=loss3_SLclassifier_1, targets=y, k=5)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        top_5_acc = tf.metrics.mean(top_5_corr)

        confusion = tf.confusion_matrix(labels=y, predictions=tf.argmax(loss3_SLclassifier_1, 1), num_classes=24)


        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            #Get only the last layers to train
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

            var_list=[FC1, FC2, FC3, loss1_fc_1, loss2_fc_1, inception_5a_5x5, inception_5a_3x3, inception_5b_3x3_reduce,
                                           inception_5b_5x5_reduce, inception_5b_1x1, inception_5b_pool_proj, inception_5b_3x3, inception_5b_5x5]


            vars_to_remove = ['inception_4c/FC/kernel:0', 'inception_4c/FC/bias:0', 'incpetion_5a/FC/kernel:0', 'incpetion_5a/FC/bias:0', 'loss1/fc_1/kernel:0', 'loss1/fc_1/bias:0', 'loss2/fc_1/kernel:0','loss2/fc_1/bias:0' ,'inception_5b/3x3_reduce_weight:0', 'inception_5b/3x3_reduce_bias:0',
                              'inception_5b/5x5_reduce_weight:0', 'inception_5b/5x5_reduce_bias:0', 'inception_5b/1x1_weight:0', 'inception_5b/1x1_bias:0', 'inception_5b/pool_proj_weight:0',
                              'inception_5b/pool_proj_bias:0', 'inception_5b/3x3_weight:0', 'inception_5b/3x3_bias:0', 'inception_5b/5x5_weight:0', 'inception_5b/5x5_bias:0', 'inception_5b/FC/kernel:0', 'inception_5b/FC/bias:0']
            trainable_variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
            #self.log.info(tf.trainable_variables())
            #self.log.info(trainable_variables)
            for var in trainable_variables:
                #self.log.info('Var name: ' + var.name)

                if var.name in vars_to_remove:
                    trainable_variables.remove(var)
                    #self.log.info("REMOVED VARIABLE: " + var.name)

            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss, var_list=trainable_variables)
            upper_train_step = tf.train.AdamOptimizer(learning_rate=.0001).minimize(total_loss, var_list=var_list)

        return loss3_SLclassifier_1, total_loss, train_step, correct_prediction, accuracy, X, y, top_5_corr, top_5_acc, confusion, upper_train_step
    def get_pictures(self, Xd, location):
        images = np.empty((Xd.shape[0], 227, 227, 3))
        #print(Xd)
        #print(location)
        for i in range(Xd.shape[0]):
            im = cv2.imread(location[i] + "/" + Xd[i])
            x = random.randint(0, 256 - 227)
            im = im[x:x+227, x:x+227]

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
                  correct_prediction=None, accuracy=None, X=None, y=None, location=None, X_val=None, y_val=None, val_loc=None,
                  saver=None, top_5_correct_prediction=None, top_5_accuracy=None, confusion=None, classes=None, upper_train_step=None):


        val_losses = []
        val_accuracies = []
        overall_train_losses = []

        relative_root = None

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [loss_val, correct_prediction, accuracy, top_5_correct_prediction, top_5_accuracy]
        if training_now:
            variables[-1] = training
            variables[2] = upper_train_step
            if self.resume:
                overall_train_losses = np.load(self.absolute_root + "/CNN/trained_networks/static_v2_lr-" + str(glr) + "/epoch-" + str(self.start_epoch) + "/static_v2_lr-" + str(glr) + "-overall_train_losses.npy").tolist()
        else:
            if self.resume:
                val_losses = np.load(self.absolute_root + "/CNN/trained_networks/static_v2_lr-" + str(glr) + "/epoch-" + str(self.start_epoch) + "/static_v2_lr-" + str(glr) + "-val_losses.npy").tolist()


        #else:
           # variables.append(confusion)


        # counter
        iter_cnt = 0

        for e in range(self.start_epoch, epochs):
            # keep track of losses and accuracy
            correct = 0
            tot_top_5_correct = 0
            losses = []
            top_5_accuracies = []
            #confusion_matricies = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                feed_dict = {X: self.get_pictures(Xd[idx], location[idx]),
                             y: yd[idx],
                             }
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                # print(actual_batch_size)confusion
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _, top_5_corr, top_5_acc = session.run(variables, feed_dict=feed_dict)
               # else:
                #    loss, corr, _, top_5_corr, top_5_acc, batch_confusion = session.run(variables, feed_dict=feed_dict)
                #    confusion_matricies.append(batch_confusion)

                # aggregate performance stats
                losses.append(loss * actual_batch_size)


                correct += np.sum(corr)
                tot_top_5_correct += np.sum(top_5_corr)
                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    self.log.info("Iteration " + str(iter_cnt) + ": with minibatch training loss = " + str(loss) + ", an accuracy of " + str(np.sum(corr) / actual_batch_size) + " and a top-5 accuracy of " + str(np.sum(top_5_corr) / actual_batch_size))



                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            epoch_loss = np.sum(losses) / Xd.shape[0]
            overall_train_losses.append(epoch_loss)
            epoch_top_5_acc = tot_top_5_correct /  Xd.shape[0]
            #val_confusion_matrix = np.sum(confusion_matricies) / Xd.shape[0]
            relative_root = self.absolute_root + "/CNN/trained_networks/static_v2_lr-" + str(glr) + "/epoch-" + str(e + 1)

            if training_now:
                self.log.info("Epoch " + str(e + 1) + ", Overall loss = " + str(epoch_loss) + ", an accuracy of " + str(total_correct) + " and a top-5 accuracy of " + str(epoch_top_5_acc))
            if plot_losses and training_now is True:
                self.log.info('Validation-------------------------')
                os.mkdir(relative_root)

                cur_val_loss, cur_val_correct = self.run_model(session, predict=predict, loss_val=loss_val, Xd=X_val, yd=y_val,
                                  epochs=self.start_epoch+1, batch_size=self.batch_size, print_every=10000, training=None, plot_losses=True,
                                  correct_prediction=correct_prediction, accuracy=accuracy, X=X, y=y, location=val_loc, top_5_correct_prediction=top_5_correct_prediction,
                                  top_5_accuracy=top_5_accuracy, confusion=confusion, classes=classes)
                val_losses.append(cur_val_loss)
                val_accuracies.append(cur_val_correct)

                saver.save(session, relative_root + "/static_v2_lr-" + str(glr) + ".ckpt")

                plt.plot(overall_train_losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('Epoch number')
                plt.ylabel('Epoch Loss')

                np.save(relative_root +"/static_v2_lr-" + str(glr) +"-overall_train_losses.npy", overall_train_losses)
                plt.savefig(relative_root +"/static_v2_lr-" + str(glr) +"-TRAIN.png")

                plt.clf()
                plt.plot(val_losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('Validation run')
                plt.ylabel('Loss')

                np.save(relative_root +"/static_v2_lr-" + str(glr) +"-val_losses.npy", val_losses)
                plt.savefig(relative_root +"/static_v2_lr-" + str(glr) + "-VAL.png")
                plt.clf()

                #plt.show()
            if not training_now:
                self.log.info("Validation, Overall loss = " + str(epoch_loss) + ", an accuracy of " + str(total_correct)+ " and a top-5 accuracy of " + str(epoch_top_5_acc))
                #val_confusion_matrix = pd.DataFrame(val_confusion_matrix, columns=classes, index=classes)
                #heatmap = sns.heatmap(val_confusion_matrix, annot=True).get_figure()
                #heatmap.savefig(relative_root +"/static_v2_lr-" + str(glr) + "-confusion_matrix.png")

        return overall_train_losses, total_correct

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
                # builder = tf.saved_model.builder.SavedModelBuilder('CNN/saved_model_v2.pb')
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
        data = pd.read_csv(self.absolute_root + '/227X227_v2.csv')

        X_data = np.array(data['file_name'])
        y_data = np.array(data['Letter'])
        location = np.array(data['dir_name'])

        label_encoder = LabelEncoder()
        #'Part5', 'Part2', 'Part3', 'part4','Part1'
        train_signers = ['C', 'B', 'D', 'A', 'Part5', 'Part2', 'Part3', 'part4', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_9', 'user_10', 'Part1']
        test_signers = ['E']

        X_train = data.loc[data['dir_name'].isin(train_signers)]['file_name'].values
        y_train = data.loc[data['dir_name'].isin(train_signers)]['Letter'].values
        y_train = label_encoder.fit_transform(y_train)
        train_loc = data.loc[data['dir_name'].isin(train_signers)]['root'].values

        X_val = data.loc[data['dir_name'].isin(test_signers)]['file_name'].values
        y_val = data.loc[data['dir_name'].isin(test_signers)]['Letter'].values
        y_val = label_encoder.fit_transform(y_val)
        val_loc = data.loc[data['dir_name'].isin(test_signers)]['root'].values

        #print(X_train)
        # X_train = np.concatenate((X_data[:26445], X_data[39838:67209]))
        #
        # label_encoder = LabelEncoder()
        # y_train = label_encoder.fit_transform(np.concatenate((y_data[:26445], y_data[39838:67209])))
        # train_loc = np.concatenate((location[:26445], location[39838:67209]))
        #
        # #65773
        # X_val = np.concatenate((X_data[26445:39838], X_data[67209:]))
        # y_val = label_encoder.fit_transform(np.concatenate((y_data[26445:39838], y_data[67209:])))
        # val_loc = np.concatenate((location[26445:39838],location[67209:]))

        for lr in self.learning_rate:
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

                with tf.device("/device:GPU:0"):  # "/cpu:0" or "/gpu:0"
                    global glr
                    relative_root = self.absolute_root + "/CNN/trained_networks/static_v2_lr-" + str(lr)

                    glr = lr
                    if self.resume is False:
                        os.mkdir(relative_root)
                    if self.resume and os.path.isdir(relative_root) is False:
                        os.mkdir(relative_root)

                    fileh = logging.FileHandler(relative_root + "/static_v2_lr-" + str(lr) + ".txt", "a+")
                    fileh.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
                    fileh.setLevel(logging.INFO)

                    self.log.handlers = [fileh]
                    #log.addHandler(fileh)
                    #log.info('test')

                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

                        loss3_SLclassifier_1, total_loss, train_step, correct_prediction, accuracy, X, y, top_5_corr, top_5_acc, confusion, upper_train_step = self.create_graph(lr, mode="training")

                        #saver = tf.train.Saver()

                        #sess.run(tf.local_variables_initializer())

                        if self.resume:
                            self.log.info("ENTERED RESUME")
                            #saver.restore(sess=sess, save_path=relative_root + "/epoch-20/static_v2_lr-" + str(lr) + ".ckpt")
                            #chkp.print_tensors_in_checkpoint_file(relative_root + "/epoch-20/static_v2_lr-" + str(lr) + ".ckpt", tensor_name='', all_tensors=True)

                            reader = tf.train.NewCheckpointReader(relative_root + "/epoch-20/static_v2_lr-" + str(lr) + ".ckpt")
                            restore_dict = dict()
                            for v in tf.trainable_variables():
                              tensor_name = v.name.split(':')[0]
                              if reader.has_tensor(tensor_name):
                                self.log.info('has tensor ', tensor_name)
                                restore_dict[tensor_name] = v

                            saver = tf.train.Saver(restore_dict)
                            sess.run(tf.local_variables_initializer())
                            saver.restore(sess=sess, save_path=relative_root + "/epoch-20/static_v2_lr-" + str(lr) + ".ckpt")

                            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in tf.global_variables()])
                            not_initialized_vars = [v for (v, f) in zip(tf.global_variables(), is_not_initialized) if not f]
                            self.log.info("NON INITIALIZED VARIABLES" + str(not_initialized_vars))
                            sess.run(tf.initialize_variables(not_initialized_vars))

                        else:
                            sess.run(tf.global_variables_initializer())

                        self.log.info('Training')

                        self.run_model(sess, predict=loss3_SLclassifier_1, loss_val=total_loss, Xd=X_train, yd=y_train,
                                  epochs=self.start_epoch + 50, batch_size=self.batch_size, print_every=100, training=train_step, plot_losses=True,
                                  correct_prediction=correct_prediction, accuracy=accuracy, X=X, y=y, location=train_loc, X_val=X_val, y_val=y_val, val_loc=val_loc, saver=saver,
                                  top_5_correct_prediction=top_5_corr, top_5_accuracy=top_5_acc, confusion=confusion, classes=list(label_encoder.classes_), upper_train_step=upper_train_step)

                        #self.run_model(sess, loss3_SLclassifier_1, total_loss, X_train, y_train, 500, 128, 100, train_step, True, correct_prediction, accuracy, X, y, train_loc)

                        #saver.save(sess, relative_root + "/final/static_v2_lr-" + str(lr) + ".ckpt")

                        #self.log.info('Validation')
                        #loss3_SLclassifier_1, total_loss, train_step, correct_prediction, accuracy, X, y = create_graph(lr, False)
                        #scope.reuse_variables()
