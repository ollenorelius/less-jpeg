import tensorflow as tf
import numpy as np
import params as p

def create_loss(input_tensor, prediction):

    tf.summary.histogram('input_tensor_loss', input_tensor)
    tf.summary.histogram('Prediction_loss', prediction)

    horz_diff = get_horz_grad(input_tensor) - get_horz_grad(prediction)
    vert_diff = get_vert_grad(input_tensor) - get_vert_grad(prediction)
    diag_diff = get_diag_grad(input_tensor) - get_diag_grad(prediction)

    grad_sum = tf.reduce_sum(tf.pow(horz_diff,2)
                            + tf.pow(vert_diff,2)
                            + tf.pow(diag_diff,2))

    abs_sum = tf.reduce_sum(tf.pow(input_tensor - prediction,2))

    a = 0.5 #weighting for absolute vs edge
    return abs_sum
    #return (a*abs_sum + grad_sum)/(p.BATCH_SIZE*p.IMAGE_SIZE*p.IMAGE_SIZE*p.CHANNELS)

def get_horz_grad(tensor):
    inc = int(tensor.get_shape()[3])
    w_h = np.tile([-1,0,1], [1,1,inc,1])
    return tf.nn.conv2d(tensor,w_h,strides=[1,1,1,1], padding='SAME')

def get_vert_grad(tensor):
    inc = int(tensor.get_shape()[3])
    w_v = np.tile(np.transpose([-1,0,1]), [1,1,inc,1])
    return tf.nn.conv2d(tensor,w_v,strides=[1,1,1,1], padding='SAME')

def get_diag_grad(tensor):
    return get_vert_grad(get_horz_grad(tensor))
