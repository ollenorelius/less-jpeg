import tensorflow as tf
import network as net
from input_producer import InputProducer
import loss

data_folder = 'data'
input_prod = InputProducer(data_folder)

input_batch = tf.placeholder(tf.float32)

input_batch_compressed = InputProducer.compress_batch(input_batch, 35)

feature_representation = net.create_forward_net(input_batch_compressed)

return_image = net.create_backward_net(feature_representation, input_batch)

loss_value = loss.create_loss(input_batch, return_image)
