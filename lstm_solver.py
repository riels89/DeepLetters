import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging
from matplotlib import pyplot as plt
import math
import sys
import os

resume = None

log = logging.getLogger()
log.setLevel(logging.DEBUG)

def get_pool_layers(locations):
    return [np.squeeze(np.load(location)) for location in locations]

def run_model(session, predict, loss_val, locations, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False,
              correct_prediction=None, accuracy=None, X=None, y=None, lr=None,
              source_seq_length=None, target_sequence_length=None, actual_batch_size=None):


    # shuffle indicies
    train_indicies = np.arange(locations.shape[0])
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
        for i in range(int(math.ceil(locations.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % locations.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            videos = get_pool_layers(locations[idx])
            source_lengths = [video.shape[0] for video in videos]
            target_lengths = [len(target) for target in yd[idx]]

            padded_videos = np.zeros([len(source_lengths), max(source_lengths), videos[0].shape[1]])
            padded_yd = np.zeros([len(target_lengths), max(target_lengths)])

            for video in videos:
                padded_videos[:, :video.shape[0], :] = video

            for word in yd[idx]:
                padded_yd[:, :len(word)] = word

            feed_dict = {X: padded_videos,
                         y: padded_yd,
                         source_seq_length: source_lengths,
                         target_sequence_length: target_lengths,
                         actual_batch_size: padded_yd.shape[0]}

            for video in feed_dict[X]:
                print(video.shape)
            print(padded_yd.shape)
            # get batch size
            real_batch_size = yd[idx].shape[0]

            # print(actual_batch_size)
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                log.info("Iteration " + str(iter_cnt) + ": with minibatch training loss = " + str(loss) + " and accuracy of " + str(np.sum(corr) / actual_batch_size))

            iter_cnt += 1
        total_correct = correct / locations.shape[0]
        epoch_loss = np.sum(losses) / locations.shape[0]
        log.info("Epoch " + str(e + 1) + ", Overall loss = " + str(epoch_loss) + " and accuracy of " + str(total_correct))
        if plot_losses and e == epochs - 1:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            if training_now is True:
                plt.savefig("CNN/trained_networks/static_v1_lr-" + str(lr) +"/static_v1_lr-" + str(lr) +"-TRAIN.png")
            else:
                plt.savefig("CNN/trained_networks/static_v1_lr-" + str(lr) +"/static_v1_lr-" + str(lr) + "-VAL.png")
            #plt.show()
    return epoch_loss, total_correct


def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.float32, [None, None, 1024], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')

    source_seq_length = tf.placeholder(tf.int32, [None], name='source_seq_length')

    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)

    batch_size = tf.placeholder(tf.int32, (), name='actual_batch_size')

    return inputs, targets, target_sequence_length, max_target_len, source_seq_length, batch_size


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    # get '<GO>' id
    go_id = tf.convert_to_tensor(target_vocab_to_int['<GO>'], tf.int32)

    # after_slice = tf.strided_slice(target_data, [0, 0], [-1, -1], [1, 1])

    after_concat = tf.concat([tf.fill([tf.cast(batch_size, tf.int32), 1], go_id), target_data], 1)

    return after_concat



def encoding_layer(rnn_inputs, rnn_size, keep_prob, source_seq_length):
    """
    :return: tuple (RNN output, RNN state)
    """

    lstm_cells = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob)
    #lstm_cells = tf.contrib.rnn.LSTMCell(rnn_size)

    outputs, state = tf.nn.dynamic_rnn(lstm_cells,
                                       rnn_inputs,
                                       sequence_length=source_seq_length,
                                       dtype=tf.float32)
    return outputs, state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                               target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_length)
    return outputs

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                      tf.cast(tf.fill([batch_size], start_of_sequence_id), tf.int32),
                                                      tf.cast(end_of_sequence_id, tf.int32))

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True)
    return outputs

def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    cells = tf.contrib.rnn.LSTMCell(rnn_size)

    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state,
                                            cells,
                                            dec_embed_input,
                                            target_sequence_length,
                                            max_target_sequence_length,
                                            output_layer,
                                            keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state,
                                            cells,
                                            dec_embeddings,
                                            target_vocab_to_int['<GO>'],
                                            target_vocab_to_int['<EOS>'],
                                            max_target_sequence_length,
                                            target_vocab_size,
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output)


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  max_target_sentence_length,
                  target_vocab_size,
                  dec_embedding_size,
                  rnn_size, target_vocab_to_int, target_sequence_length, source_seq_length):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data,
                                             rnn_size,
                                             keep_prob,
                                             source_seq_length)

    dec_input = process_decoder_input(target_data,
                                      target_vocab_to_int,
                                      batch_size)

    train_output, infer_output = decoding_layer(dec_input,
                                                enc_states,
                                                target_sequence_length,
                                                max_target_sentence_length,
                                                rnn_size,
                                                target_vocab_to_int,
                                                target_vocab_size,
                                                batch_size,
                                                keep_prob,
                                                dec_embedding_size)

    return train_output, infer_output


def init_constants():
    letters = list('qwertyuiopasdfghjklzxcvbnm')
    letters.append('<GO>')
    letters.append('<EOS>')
    data = pd.read_csv('pool_layers.csv')
    words = data['word']

    le = LabelEncoder()
    le.fit(letters)
    target_vocab_to_int = dict(zip(le.classes_, le.transform(le.classes_)))
    data['word'] = [le.transform(list(word)) for word in words]

    train_signers = ['signer 1']
    test_signers = ['signer 1']

    loc_train = data.loc[data['signer'].isin(train_signers)]['filepath']
    words_train = data.loc[data['signer'].isin(train_signers)]['word'].values

    print(loc_train)
    print(words_train)

    loc_val = data.loc[data['signer'].isin(test_signers)]['filepath']
    words_val = data.loc[data['signer'].isin(test_signers)]['word']

    target_vocab_size = 26

    decoding_embedding_size = 100
    lrs = [.001]
    keep_prob = 0.2
    rnn_size = 1024

    return target_vocab_to_int, loc_train, words_train, loc_val, words_val, target_vocab_size, decoding_embedding_size, lrs, keep_prob, rnn_size

def create_graph(target_vocab_to_int, keep_prob, target_vocab_size, decoding_embedding_size, rnn_size, lr):
    input_data, targets, target_sequence_length, max_target_sequence_length, source_seq_length, batch_size = enc_dec_model_inputs()

    train_logits, inference_logits = seq2seq_model(input_data=tf.reverse(input_data, [-1]),
                                                   target_data=targets,
                                                   keep_prob=keep_prob,
                                                   batch_size=batch_size,
                                                   max_target_sentence_length=max_target_sequence_length,
                                                   target_vocab_size=target_vocab_size,
                                                   dec_embedding_size=decoding_embedding_size,
                                                   rnn_size=rnn_size,
                                                   target_vocab_to_int=target_vocab_to_int,
                                                   target_sequence_length=target_sequence_length,
                                                   source_seq_length=source_seq_length)

    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    accuracy = tf.contrib.metrics.accuracy(inference_logits, targets)

    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    return inference_logits, accuracy, cost, input_data, targets, train_op, source_seq_length, target_sequence_length, batch_size

if __name__ ==  "__main__":

    resume = sys.argv[1]
    #print(resume is False)
    train_graph = tf.Graph()
    with tf.Session(graph=train_graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        with tf.device("/device:GPU:0"):  # "/cpu:0" or "/gpu:0"

            target_vocab_to_int, loc_train, words_train, loc_val, words_val, target_vocab_size, \
                decoding_embedding_size, lrs, keep_prob, rnn_size = init_constants()

            for lr in lrs:
                relative_root = "CNN/trained_networks/lstm_v1_lr-" + str(lr)

                if resume is False:
                    os.mkdir(relative_root)
                if resume and os.path.isdir(relative_root) is False:
                    os.mkdir(relative_root)

                fileh = logging.FileHandler(relative_root + "/lstm_v1_lr-" + str(lr) + ".txt", "a+")
                fileh.setFormatter(logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s"))
                fileh.setLevel(logging.DEBUG)

                inference_logits, accuracy, cost, input_data, targets, train_op, source_seq_length, target_sequence_length, batch_size = create_graph(target_vocab_to_int, keep_prob, target_vocab_size, decoding_embedding_size, rnn_size, lr)

                saver = tf.train.Saver()

                sess.run(tf.local_variables_initializer())

                if resume and os.path.exists(relative_root + "/lstm_v1_lr-" + str(lr) + ".ckpt"):
                    saver.restore(sess=sess, save_path=relative_root + "/lstm_v1_lr-" + str(lr) + ".ckpt")
                else:
                    sess.run(tf.global_variables_initializer())

                saver.save(sess, relative_root + "/lstm_v1_lr-" + str(lr) + ".ckpt")

                log.info('Training')

                run_model(sess, predict=inference_logits, loss_val=cost, locations=loc_train, yd=words_train,
                          epochs=100, batch_size=64, print_every=100, training=train_op, plot_losses=True,
                          correct_prediction=inference_logits, accuracy=accuracy, X=input_data, y=targets,
                          source_seq_length=source_seq_length, target_sequence_length=target_sequence_length, actual_batch_size=batch_size)

                saver.save(sess, relative_root + "/lstm_v1_lr-" + str(lr) + ".ckpt")

                log.info('Validation')

                run_model(sess, predict=inference_logits, loss_val=cost, locations=loc_val, yd=words_val,
                          epochs=100, batch_size=64, print_every=100, training=None, plot_losses=True,
                          correct_prediction=inference_logits, accuracy=accuracy, X=input_data, y=targets,
                          source_seq_length=source_seq_length, target_sequence_length=target_sequence_length, actual_batch_size=batch_size)




