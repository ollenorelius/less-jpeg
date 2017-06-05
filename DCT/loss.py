"""Contains everything related to loss."""
import tensorflow as tf


def create_loss(input_tensor, prediction):
    """Construct loss tensor."""
    tf.summary.histogram('input_tensor_loss', input_tensor)
    tf.summary.histogram('Prediction_loss', prediction)

    abs_sum = tf.reduce_mean(tf.abs(input_tensor - prediction))

    return abs_sum
