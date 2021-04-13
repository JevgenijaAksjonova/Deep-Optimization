import tensorflow.compat.v1 as tf
import tensorflow.contrib as tc

FILTERS = 32
KSIZE = 3
MAX_FILTERS = 512
N_OUT = 1

def apply_conv(x, filters=FILTERS, kernel_size=KSIZE):
    initializer = tc.layers.variance_scaling_initializer()
    kernel_regularizer = tc.layers.l2_regularizer(scale=1e-5)

    return tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                            padding='SAME',
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer)

def activation(x):
    with tf.name_scope('activation'):
        return tf.nn.leaky_relu(x)

def inst_norm(x):
    return tc.layers.instance_norm(x) 

def convnet(inp, n_layers, n_out=N_OUT, filters=FILTERS):
    with tf.name_scope('convnet'):
        x = inst_norm(inp)
        for i in range(n_layers):
            x = apply_conv(x, filters = filters)
            x = inst_norm(x)
            x = activation(x)
        result = tf.layers.conv2d(x, filters=n_out, kernel_size=KSIZE, padding='SAME')
    return result
