import tensorflow as tf
import params as p

def create_forward_net(input_tensor):
    with tf.name_scope('Forward_net'):
        l1 = conv_layer(input_tensor, 5, 18, 1, 'layer_1')
        s2 = conv_layer(l1, 5, 18, 2, 'shrink_layer_2') #128
        l3 = conv_layer(s2, 3, 36, 1, 'layer_3')
        s5 = conv_layer(l3, 3, 36, 2, 'shrink_layer_5') # 64
        l6 = conv_layer(s5, 3, 36, 1, 'layer_9')
        return l6

def create_backward_net(input_tensor, original_batch):
    with tf.name_scope('Backward_net'):

        r_l6 = r_conv_layer(input_tensor, 3, 36, 1, 'r_layer_9')
        r_s5 = r_conv_layer(r_l6, 3, 36, 2, 'r_shrink_layer_5') # 128
        r_l3 = r_conv_layer(r_s5, 3, 36, 1, 'r_layer_3')
        r_s2 = r_conv_layer(r_l3, 3, 64, 2, 'r_shrink_layer_2') #256
        r_l4 = r_conv_layer(r_s2, 5, 64, 1, 'r_layer_4')

        r_l1 = output_layer(r_l4, 1, p.CHANNELS, 1, 'r_layer_1')

        return r_l1# + original_batchd

def create_deep_forward_net(input_tensor):
    with tf.name_scope('Forward_net'):
        l1 = conv_layer(input_tensor, 5, 18, 1, 'layer_1')
        s2 = conv_layer(l1, 5, 18, 2, 'shrink_layer_2') #128
        l3 = conv_layer(s2, 3, 36, 1, 'layer_3')
        s5 = conv_layer(l3, 3, 36, 2, 'shrink_layer_5') # 64
        l6 = conv_layer(s5, 3, 36, 1, 'layer_9')
        l7 = conv_layer(l6, 3, 48, 1, 'layer_9')
        s6 = conv_layer(l7, 3, 48, 2, 'shrink_layer_5') # 32
        l8 = conv_layer(s6, 3, 80, 1, 'layer_9')
        l9 = conv_layer(l8, 3, 80, 1, 'layer_9')
        return l9

def create_deep_backward_net(input_tensor, original_batch):
    with tf.name_scope('Backward_net'):
        r_s5 = r_conv_layer(input_tensor, 3, 80, 2, 'r_shrink_layer_5') # 64
        r_l6 = r_conv_layer(r_s5, 3, 64, 1, 'r_layer_9')
        r_s4 = r_conv_layer(r_l6, 3, 64, 2, 'r_shrink_layer_5') # 128
        r_l3 = r_conv_layer(r_s4, 3, 64, 1, 'r_layer_3')
        r_s2 = r_conv_layer(r_l3, 3, 64, 2, 'r_shrink_layer_2') #256
        r_l4 = r_conv_layer(r_s2, 5, 64, 1, 'r_layer_4')

        r_l1 = output_layer(r_l4, 1, p.CHANNELS, 1, 'r_layer_1')

        return r_l1# + original_batch



def output_layer(input_tensor, kernel_size, depth, stride, name):
    with tf.name_scope('output_layer'):
        inc = int(input_tensor.get_shape()[3])
        w = weight_variable([kernel_size,kernel_size,inc,depth],'w_conv')
        b = bias_variable([depth],'b_conv')
        c = conv2d(input_tensor, w, stride) + b
        return c

def conv_layer(input_tensor, kernel_size, depth, stride, name):
    with tf.name_scope(name):
        inc = int(input_tensor.get_shape()[3])
        w = weight_variable([kernel_size,kernel_size,inc,depth],'w_conv')
        b = bias_variable([depth],'b_conv')
        c = layer_activation(conv2d(input_tensor, w, stride) + b)
        return c

def r_conv_layer(input_tensor, kernel_size, depth, stride, name):
    with tf.name_scope(name):
        inc = int(input_tensor.get_shape()[3])
        #batch_size = int(input_tensor.get_shape()[0])
        w = weight_variable([kernel_size,kernel_size,depth,inc],'w_conv')
        b = bias_variable([depth],'b_conv')

        dyn_input_shape = tf.shape(input_tensor)
        batch_size = dyn_input_shape[0]

        out_shape = [batch_size, dyn_input_shape[1]*stride,
        dyn_input_shape[2]*stride, depth]

        c = layer_activation(t_conv2d(input_tensor, w, out_shape, stride) + b)
        return c

def r_conv_upsample(input_tensor, kernel_size, depth, name):
    with tf.name_scope(name):
        inc = int(input_tensor.get_shape()[3])
        w = weight_variable([kernel_size,kernel_size,depth,inc],'w_conv')
        b = bias_variable([depth],'b_conv')

        out_shape = tf.constant([input_tensor.get_shape()[0],
        input_tensor.get_shape()[1]*stride,
        input_tensor.get_shape()[2]*stride,
        depth])

        c = layer_activation(t_conv2d(input_tensor, w, out_shape, stride) + b)
        return c

def layer_activation(input_tensor):
    '''
    Convenience function for trying different activations.
    '''
    return tf.nn.relu(input_tensor)

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x,W, stride):
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1], padding='SAME')

def t_conv2d(x,W, out_shape, stride):
    return tf.nn.conv2d_transpose(x,W,out_shape,strides=[1,stride,stride,1], padding='SAME')
