import tensorflow as tf

def create_forward_net(input_tensor):
    with tf.name.scope('Forward net'):
        l1 = conv_layer(input_tensor, 5, 64, 1, 'layer_1')
        s2 = conv_layer(input_tensor, 5, 64, 2, 'shrink_layer_2') #128

        l3 = conv_layer(input_tensor, 5, 96, 1, 'layer_3')
        l4 = conv_layer(input_tensor, 5, 96, 1, 'layer_4')
        s5 = conv_layer(input_tensor, 5, 96, 2, 'shrink_layer_5') # 64

        l6 = conv_layer(input_tensor, 5, 128, 1, 'layer_6')
        l7 = conv_layer(input_tensor, 5, 128, 1, 'layer_7')
        s8 = conv_layer(input_tensor, 5, 128, 2, 'shrink_layer_8') # 32

        l9 = conv_layer(input_tensor, 3, 128, 1, 'layer_9')

        return l9

def create_backward_net(input_tensor, original_batch):
    with tf.name.scope('Backward net'):

        r_l9 = r_conv_layer(input_tensor, 3, 128, 1, 'r_layer_9')
        r_s8 = r_conv_layer(input_tensor, 5, 128, 2, 'r_shrink_layer_8') # 64

        r_l7 = r_conv_layer(input_tensor, 5, 128, 1, 'r_layer_7')
        r_l6 = r_conv_layer(input_tensor, 5, 128, 1, 'r_layer_6')

        r_s5 = r_conv_layer(input_tensor, 5, 96, 2, 'r_shrink_layer_5') # 128
        r_l4 = r_conv_layer(input_tensor, 5, 96, 1, 'r_layer_4')

        r_l3 = r_conv_layer(input_tensor, 5, 96, 1, 'r_layer_3')
        r_s2 = r_conv_layer(input_tensor, 5, 64, 2, 'r_shrink_layer_2') #256

        r_l1 = outputlayer(input_tensor, 1, 3, 1, 'r_layer_1')


        return r_l1 + original_batch



def output_layer(input_tensor):
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
        w = weight_variable([kernel_size,kernel_size,inc,depth],'w_conv')
        b = bias_variable([depth],'b_conv')

        out_shape = tf.shape(input_tensor)
        out_shape[1] = out_shape[1]*stride
        out_shape[2] = out_shape[2]*stride

        c = layer_activation(t_conv2d(input_tensor, w, stride, out_shape) + b)
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
