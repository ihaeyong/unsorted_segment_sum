import tensorflow as tf

with tf.Session() as sess:
    data = tf.constant([[0.2, 0.1, 0.5, 0.7, 0.8],
                        [0.2, 0.2, 0.5, 0.7, 0.8]])
    idx = tf.constant([0, 0, 1, 2, 2])
    ones = tf.ones_like(data)
    count = tf.transpose(tf.unsorted_segment_sum(tf.transpose(ones), idx, 3))
    result = tf.transpose(tf.unsorted_segment_sum(tf.transpose(data), idx, 3))
    print(sess.run(count))
    print(sess.run(result))
    print(sess.run(result/count))