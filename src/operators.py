import numpy as np
import tensorflow.compat.v1 as tf
import odl

# Define ODL operators

def operators_smooth():
    size = 512
    space = odl.uniform_discr([-256, -256], [256, 256], [size, size], dtype='float32', weighting=1.0)
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
    detector_partition = odl.uniform_partition(-360, 360, 1000)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    T = odl.tomo.RayTransform(space, geometry)
    fbp = odl.tomo.fbp_op(T, frequency_scaling=0.45, filter_type='Hann')
    T_norm = T.norm(estimate=True)
    T = (1 / T_norm) * T
    W = odl.Gradient(space)
    return [T, W]

def operators_nonsmooth():
    size = 512
    space = odl.uniform_discr([-256, -256], [256, 256], [size, size], dtype='float32', weighting=1.0)
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
    detector_partition = odl.uniform_partition(-360, 360, 1000)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    T = odl.tomo.RayTransform(space, geometry)
    fbp = odl.tomo.fbp_op(T, frequency_scaling=0.45, filter_type='Hann')
    T_norm = T.norm(estimate=True)
    T = (1 / T_norm) * T
    # Wavelet transform
    W = odl.trafos.wavelet.WaveletTransform(space, wavelet='sym5', nlevels=5)
    scales = W.scales()
    W = np.power(1.8, scales) * W
    return [T, W]


# Define a proximal operator

def prox_l1(x, alpha):
    res = tf.sign(x) * tf.maximum(tf.abs(x) - alpha, 0)
    return res

def prox(x, W, W_adj, gamma, lam, mu):
    y = W(x)
    alpha = lam * mu * gamma
    res = x + 1 / mu * W_adj(prox_l1(y, alpha) - y)
    return res