import tensorflow as tf
import numpy as np

def create_loss(input_tensor, prediction):
    #TODO: The point of this function is to convolve out the edges, but i'm tired
    # and so it's just regular dumb l2 for now.

    return tf.nn.l2_loss(input_tensor - prediction)
