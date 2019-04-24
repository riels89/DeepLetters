import tensorflow as tf


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

    with tf.device("/device:GPU:0"):  # "/cpu:0" or "/gpu:0"

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

            saver = tf.train.import_meta_graph("C:/Users/riley/DeepLetters/trained_model/static_v2_lr-1e-06.ckpt.meta")

            saver.restore(sess=sess, save_path="C:/Users/riley/DeepLetters/trained_model/static_v2_lr-1e-06.ckpt")

            graph = sess.graph
            probs = graph.get_tensor_by_name('prob3:0')

            output_node_names = ['prob3', 'prob2', 'prob1']

            frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph_def,
                output_node_names)

            # Save the frozen graph
            with open('CNN/best_cnn.pb', 'wb') as f:
              f.write(frozen_graph_def.SerializeToString())

