import tensorflow as tf

with tf.Session(graph=tf.Graph()) as sess:
   tf.saved_model.loader.load(
       sess,
       [tf.saved_model.tag_constants.SERVING],
       path_to_model)
     