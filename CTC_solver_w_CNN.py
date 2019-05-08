from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.initializers import *
from keras.optimizers import *
import tensorflow as tf
import keras.backend as kback
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
import math
from matplotlib import pyplot as plt
import pandas as pd
import os
from keras.utils import to_categorical
import logging
import numpy as np
import itertools
import keras
import sys
import cv2
import random
import gc

sys.path.append('LSTM')
import DL_pool_CNN as kit_model

mean_image = np.load('CNN/mean_image.npy')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # print(y_pred.shape)
    # print(tf.argmax(labels, axis=2).shape)
    # print(input_length.shape)
    # print(label_length.shape)

    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    return K.ctc_batch_cost(tf.argmax(labels, axis=2), y_pred, input_length, label_length)

def get_video(location):

    cap = cv2.VideoCapture('video_data/' + location + '.mp4')
    video = np.empty([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 227, 227, 3])
    fliped = random.random() < .5
    for i in range(video.shape[0]):
        ret, frame = cap.read()
        x = random.randint(0, 256 - 227)
        resized_image = frame[x:x+227, x:x+227]

        resized_image = resized_image - mean_image

        if fliped:
            resized_image = cv2.flip(resized_image, 0)

        video[i] = resized_image

    cap.release()
    cv2.destroyAllWindows()
    assert not np.any(np.isnan(video))

    return video


class lstm():

    def __init__(self, lrs, resume, start_epoch, epochs):

        self.lr = lrs
        self.resume = resume
        self.start_epoch = start_epoch
        self.epochs = epochs

        self.batch_size = 1  # Batch size for training.
        self.latent_dim = 2028  # Latent dimensionality of the encoding space.

        self.log = logging.getLogger()
        self.log.setLevel(logging.DEBUG)

        letters = list('qwertyuiopasdfghjklzxcvbnm &.@')
        letters.append('\'')
        letters.append('<GO>')
        letters.append('<EOS>')
        data = pd.read_csv('ChicagoFS/ChicagoFSWild - OG backup.csv')

        data = data.drop(data[(data.number_of_frames < data.label_proc.str.len() + 3)].index)
        data = data.drop(data[(data.number_of_frames > 110)].index)

        words = data['label_proc']

        le = LabelEncoder()
        le.fit(letters)
        self.target_vocab_to_int = dict(zip(le.classes_, le.transform(le.classes_)))
        self.int_to_vocab = dict(zip(le.transform(le.classes_), le.classes_))
        self.int_to_vocab[len(self.int_to_vocab)] = 'blank'
        print(self.int_to_vocab)
        data['label_proc'] = [le.transform(['<GO>'] + list(word) + ['<EOS>']) for word in words]

        self.target_vocab_size = len(self.int_to_vocab)

        train_signers = ['train']
        test_signers = ['dev']

        self.loc_train = data.loc[data['partition'].isin(train_signers)]['filename'].values
        self.dec_input_train = data.loc[data['partition'].isin(train_signers)]['label_proc'].values

        self.loc_val = data.loc[data['partition'].isin(test_signers)]['filename'].values
        self.dec_input_val = data.loc[data['partition'].isin(test_signers)]['label_proc'].values

        both = np.concatenate((self.dec_input_train, self.dec_input_val))

        target_lengths = [len(target) for target in both]
        self.max_seq_length = max(target_lengths)

        padded_train = np.full([len(self.dec_input_train), max(target_lengths)], 4)
        padded_val = np.full([len(self.dec_input_val), max(target_lengths)], 4)

        for word in range(len(self.dec_input_train)):
            padded_train[word, :len(self.dec_input_train[word])] = self.dec_input_train[word]

        for word in range(len(self.dec_input_val)):
                padded_val[word, :len(self.dec_input_val[word])] = self.dec_input_val[word]

        self.dec_input_train = to_categorical(padded_train, num_classes=self.target_vocab_size)

        self.dec_input_val = to_categorical(padded_val, num_classes=self.target_vocab_size)

        print("Initilized data...")

    def get_pool_layers(self, locations):
        return [np.squeeze(np.load(location)) for location in locations]

    def separate(self, word):
        arr = []
        for i in word:
            if i != 4:
                arr.append(i)
            else:
                return arr


    def format_videos(self, locations, yd, index):
        # shuffle indicies
        train_indicies = np.arange(locations.shape[0])
        np.random.shuffle(train_indicies)

        start_idx = (index * self.batch_size) % locations.shape[0]
        idx = train_indicies[start_idx:start_idx + self.batch_size]

        # create a feed dictionary for this batch
        videos = []
        for loc in locations[idx]:
            videos.append(get_video(loc))

        videos = np.array(videos)
        print(videos[0].shape)

        source_lengths = [video.shape[0] for video in videos]

        padded_videos = np.zeros([len(source_lengths), max(source_lengths), videos[0].shape[1], videos[0].shape[2], videos[0].shape[3]])

        for video in videos:
            padded_videos[:, :video.shape[0], :] = video

        padded_yd = yd[idx]

        actual_batch_size = padded_yd.shape[0]

        source_target_len = np.array([len(self.separate(np.argmax(padded_yd, axis=2)[0]))+1 for y in np.argmax(padded_yd, axis=2)])
        # print("Target Len:" + str(source_target_len))
        # print("Joined:" + str(self.separate(np.argmax(padded_yd, axis=2)[0])))
        # print("Padded Y:" + str(np.argmax(padded_yd, axis=2)))

        return padded_videos, padded_yd, actual_batch_size, np.array(source_lengths), source_target_len

    def get_model(self, lr):

        inputs = Input(shape=(None, 227, 227, 3))
        labels = Input(shape=(None, self.target_vocab_size))
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        with tf.device('/gpu:0'):

            final_pool = TimeDistributed(kit_model.KitModel('LSTM/weights_w_G.npy'))(inputs)

        with tf.device('/cpu:0'):

            rnn = GRU(self.latent_dim, return_sequences=True)

            encoder_outputs = rnn(final_pool)

            out_dense = (Dense(self.target_vocab_size, activation='softmax', name='softmax'))

            output = out_dense(encoder_outputs)

            loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([output, labels, input_length, label_length])

            model = Model([inputs, labels, input_length, label_length], loss_out)

            model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=0.001, clipnorm=1))

            decode_model = Model([inputs, input_length], output)

        return model, decode_model

    def decode_sequence(self, input_seq, decode_model):

        # Encode the input as state vectors.
        #states_value = model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.target_vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_vocab_to_int['<GO>']] = 1.

        predicted_seq = ""
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        output_tokens = decode_model.predict([input_seq, np.array(len(input_seq[0]))[np.newaxis]])

        print(np.argmax(np.squeeze(output_tokens), axis=1))
        print(np.max(np.squeeze(output_tokens), axis=1))

        output_tokens = K.get_value(K.ctc_decode(output_tokens, input_length=np.ones(output_tokens.shape[0])*output_tokens.shape[1],
                         greedy=True)[0][0])


            # Sample a token
        #sampled_token_index = np.argmax(output_tokens, axis=1)
        #print(output_tokens_k)
        for t in range(len(output_tokens[0])):
            predicted_seq += self.int_to_vocab[output_tokens[0][t]]
        print(predicted_seq)

        return predicted_seq

    def write_log(self, callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()


    def run_model(self, lr=None, print_every=100):


        val_losses = []
        overall_train_losses = []
        overall_val_losses = []


        if self.resume:
            #overall_train_losses = np.load(self.absolute_root + "/CNN/trained_networks/static_v2_lr-" + str(glr) + "/epoch-" + str(self.start_epoch) + "/static_v2_lr-" + str(glr) + "-overall_train_losses.npy").tolist()
            overall_train_losses = np.load("C:/Riley/DeepLetters/CNN/trained_networks/static_v2_lr-1e-06/epoch-" + str(self.start_epoch) + "/static_v2_lr-1e-06-overall_train_losses.npy").tolist()
            val_losses = np.load("C:/Riley/DeepLetters/CNN/trained_networks/static_v2_lr-1e-06/epoch-" + str(self.start_epoch) + "/static_v2_lr-1e-06-val_losses.npy").tolist()
            #val_losses = np.load(self.absolute_root + "/CNN/trained_networks/static_v2_lr-" + str(glr) + "/epoch-" + str(self.start_epoch) + "/static_v2_lr-" + str(glr) + "-val_losses.npy").tolist()

        iter_cnt = 0

        print("Creating model...")
        model, decode_model = self.get_model(lr)
        print("Finished creating model")

        # tensorboard = keras.callbacks.TensorBoard(
        #   log_dir="./LSTM/trained_networks/LSTM-lr-" + str(lr),
        #   histogram_freq=1,
        #   write_graph=True,
        #   write_grads=True,
        #   write_images=True
        # )
        # tensorboard.set_model(model)
        # train_names = ['train_loss']
        # val_names = ['val_loss']

        #model.summary(print_fn=self.log.info)
        for e in range(self.start_epoch, self.epochs):
            losses = []
            for i in range(int(math.ceil(self.loc_train.shape[0] / self.batch_size))):
                padded_videos, padded_yd, actual_batch_size, input_len, label_len = self.format_videos(locations=self.loc_train, yd=self.dec_input_train, index=i)
                #decoder_target_data = np.append(padded_yd[:, 1:, :], np.zeros((padded_yd.shape[0], 1, 29)), axis=1)
                # print(type(padded_videos))
                # print(type(padded_yd))
                # print(type(input_len))
                # print(type(label_len))
                loss = model.train_on_batch([padded_videos, padded_yd, input_len, label_len], np.zeros([actual_batch_size]))
                #self.write_log(tensorboard, train_names, [loss], e)
                losses.append(loss * actual_batch_size)

                if (iter_cnt % 100) == 0:
                    self.log.info("Iteration " + str(iter_cnt) + ": with minibatch training loss = " + str(loss))
                    ground_truth = [self.int_to_vocab[np.argmax(letter, axis=0)] for letter in padded_yd[0]]
                    self.log.info("Sample: " + str(self.decode_sequence(padded_videos[0].reshape(1,-1,227,227,3), decode_model)) + "Ground Truth: " + str(ground_truth))

                iter_cnt += 1
            epoch_loss = np.sum(losses) / self.loc_train.shape[0]
            overall_train_losses.append(epoch_loss)
            #val_confusion_matrix = np.sum(confusion_matricies) / Xd.shape[0]
            relative_root = "./LSTM/trained_networks/LSTM-lr-" + str(lr) + "/epoch-" + str(e + 1)

            self.log.info("Epoch " + str(e + 1) + ", Overall loss = " + str(epoch_loss))

            self.log.info('Validation-------------------------')
            os.mkdir(relative_root)

            for i in range(int(math.ceil(self.loc_train.shape[0] / self.batch_size))):
                padded_videos, padded_yd, actual_batch_size, input_len, label_len = self.format_videos(locations=self.loc_val, yd=self.dec_input_val, index=i)
                #decoder_target_data = np.append(padded_yd[:, 1:, :], np.zeros((padded_yd.shape[0], 1, 29)), axis=1)

                cur_val_loss = model.evaluate([padded_videos, padded_yd, input_len, label_len], np.zeros([actual_batch_size]))
                #self.write_log(tensorboard, val_names, [cur_val_loss], e)

                if i%4 == 0:
                    ground_truth = [self.int_to_vocab[np.argmax(letter, axis=0)] for letter in padded_yd[0]]
                    self.log.info("Sample: " + str(self.decode_sequence(padded_videos[0].reshape(1,-1,227,227,3), decode_model)) + "Ground Truth: " + str(ground_truth))

                val_losses.append(cur_val_loss * actual_batch_size)

            val_epoch_loss = np.sum(val_losses) / self.loc_train.shape[0]
            overall_val_losses.append(val_epoch_loss)
            val_losses = []
            self.log.info("Validation, Overall loss = " + str(val_epoch_loss))
            # Save Model
            file = "./LSTM/trained_networks/LSTM-lr-" + str(lr) + "/epoch-" + str(e + 1 - 5)
            if os.path.isdir(file):
                os.remove(file + '/train_model.h5')
                os.remove(file + '/decode_model.h5')

            model.save(relative_root + "/train_model.h5")
            decode_model.save(relative_root + "/decode_model.h5")

            plt.plot(overall_train_losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('Epoch number')
            plt.ylabel('Epoch Loss')

            np.save(relative_root + "/overall_train_losses.npy", overall_train_losses)
            plt.savefig(relative_root +"/TRAIN.png")

            plt.clf()
            plt.plot(overall_val_losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('Validation run')
            plt.ylabel('Loss')

            np.save(relative_root + "/val_losses.npy", overall_val_losses)
            plt.savefig(relative_root +"/VAL.png")
            plt.clf()

            #val_confusion_matrix = pd.DataFrame(val_confusion_matrix, columns=classes, index=classes)
            #heatmap = sns.heatmap(val_confusion_matrix, annot=True).get_figure()
            #heatmap.savefig(relative_root +"/static_v2_lr-" + str(glr) + "-confusion_matrix.png")

        return overall_train_losses

    def train(self):
        lr = self.lr
        relative_root = "./LSTM/trained_networks/LSTM-lr-" + str(lr)
        os.mkdir(relative_root)
        fileh = logging.FileHandler(relative_root + "/log.txt", "a+")
        fileh.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
        fileh.setLevel(logging.DEBUG)

        self.log.handlers = [fileh]
        self.run_model(lr)
