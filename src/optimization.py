import numpy as np
import tensorflow.compat.v1 as tf
import odl
from odl.contrib.tensorflow import as_tensorflow_layer

from operators import prox
from network import convnet

#################################
# Optimize |Tx-y|_2 + lam*|Wx|_1
#################################

# optimisation parameters
lam_smooth = 0.0015
lam_nonsmooth= 0.0005
gamma_n = 0.5
beta = 0.5
alpha = 0.999
omega = alpha
use_dx2 = True
Huber_param = 0.01

# helpers
def dot(x,y):
    dot = tf.reduce_sum(x * y, axis=[1,2,3], keepdims=True)
    return dot

def norm_sq(x):
    norm = tf.reduce_sum(x**2, axis=[1,2,3], keepdims=True)
    return norm

# safe normalization ensures that 
# there is no division by 0 in backpropagation
def safely_normalize(x):
    safe_sqrt = tf.sqrt(norm_sq(x) + 1)
    x = x / safe_sqrt
    return x

def safe_sqrt(x):
    safe_sqrt = tf.sqrt(tf.where(tf.equal(x, 0.0), x + 1, x))
    return tf.where(tf.equal(x, 0.0), tf.zeros_like(x), safe_sqrt)

def norm(x):
    return safe_sqrt(norm_sq(x))

# losses
def l2(x,y):
    loss = tf.reduce_mean(norm_sq(x - y))
    return loss

def l1(x):
    shape = tf.shape(x)
    x_flat = tf.reshape(x, shape=[shape[0],-1])
    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(x_flat), axis=-1))
    return loss

# Huber 
def huber_grad(x, gamma):
    return tf.where(tf.abs(x) < gamma, x / gamma, tf.sign(x))

def huber(x, gamma):
    shape = tf.shape(x)
    y = tf.where(tf.abs(x) < gamma, 
                 0.5 * x**2 / gamma, 
                 tf.abs(x) - 0.5 * gamma)
    y_flat = tf.reshape(y, shape=[shape[0],-1])
    loss = tf.reduce_mean(tf.reduce_sum(y_flat, axis=-1))
    return loss 

###############################
# Learned optimization schemes
###############################

def learned_opt_smooth(x, y, T, T_adj, W, W_adj, lam, n_iter):
    losses = [tf.zeros(tf.reshape(n_iter,(1,)), tf.float32)] * 3
    h_norm = 8.0 / Huber_param # |W_odl|^2
    
    xn = x
    xn_prev = x
    dx = tf.zeros_like(x)
    i = tf.constant(0, dtype=tf.int32)
    
    def cond(i, xn, xn_prev, dx, losses):
        return i < n_iter
    
    def body(i, xn, xn_prev, dx, losses):
        grad_f = 2 * T_adj(T(xn) - y) 
        grad_h = lam * W_adj(huber_grad(W(xn), Huber_param))
        grad = grad_f + grad_h
        with tf.variable_scope('body', reuse=tf.AUTO_REUSE):
            inp = tf.concat([xn, grad_f, grad_h, dx], axis=-1)
            dx = convnet(inp, n_layers=2)

        dx_n = safely_normalize(dx) * alpha * norm(grad)
        xn_prev = xn
        xn -= 1. / (2 + lam * h_norm) * (grad + dx_n)
        
        #evaluaete losses in each iteration
        index = tf.one_hot(i,n_iter)
        losses[0] += index * l2(T(xn), y)
        losses[1] += index * lam * huber(W(xn), Huber_param)
        losses[2] += index * tf.reshape(norm(xn - xn_prev), [])
        return i + 1, xn, xn_prev, dx_n, losses

    i, xn, xn_prev, dx, losses = \
        tf.while_loop(cond, body, [i, xn, xn_prev, dx, losses], swap_memory=True)
        
    loss_f = l2(T(xn), y)
    loss_g = lam * huber(W(xn), Huber_param)
    final_loss = loss_f + loss_g
    return [xn, final_loss, losses]

def learned_opt_nonsmooth(x, y, T, T_adj, W, W_adj, lam, mu, n_iter):
    losses = [tf.zeros(tf.reshape(n_iter,(1,)), tf.float32)] * 3
    
    xn = x
    xn_prev = x
    grad_f = 2 * T_adj(T(x) - y)
    dx1 = tf.zeros_like(x)
    dx2 = tf.zeros_like(x)
    i = tf.constant(0, dtype=tf.int32)
    
    def cond(i, xn, xn_prev, grad_f, dx1, dx2, losses):
        return i < n_iter
    
    def body(i, xn, xn_prev, grad_f, dx1, dx2, losses):
        dx1_prev = dx1
        dx2_prev = dx2
        with tf.variable_scope('body', reuse=tf.AUTO_REUSE):
            inp1 = tf.concat([xn, grad_f, dx1], axis=-1)
            dx1 = convnet(inp1, n_layers=2)
            if use_dx2:
                inp2 = tf.concat([xn, grad_f, dx2, dx1], axis=-1)   
                dx2 = convnet(inp2, n_layers=2)
    
        # normalize network output
        c1 = np.sqrt(2 * beta * alpha * 
                     (2 * beta - gamma_n) / 
                     (2 * beta * gamma_n))
        c1 *=  norm(xn - xn_prev - 
                    beta / (2 * beta - gamma_n) * dx2_prev) 
        #c1 = tf.stop_gradient(c1)
        dx1_n = safely_normalize(dx1) * c1 
        yn = xn + dx1_n
        grad_f_prev = grad_f
        grad_f = 2 * T_adj(T(yn) - y)
        if use_dx2:
            c2 = np.sqrt(2 * gamma_n * 
                         (2 * beta - gamma_n) / 
                         beta * beta * omega / 2)
            c2 *= norm(grad_f - grad_f_prev - 
                       1 / beta * (xn - xn_prev - dx1_prev))
            #c2 = tf.stop_gradient(c2)
            dx2_n = safely_normalize(dx2) * c2
            
        # update xn
        xn_prev = xn
        xn -= gamma_n * grad_f
        xn += gamma_n / beta * dx1_n
        if use_dx2:
            xn += dx2_n
        else:
            dx2_n = dx2_prev
        xn = prox(xn, W, W_adj, gamma_n, lam, mu)
        
        # evaluaete losses in each iteration
        index = tf.one_hot(i, n_iter)
        losses[0] += index * l2(T(xn), y)
        losses[1] += index * lam * l1(W(xn))
        Wn = (l2(T(yn), y) + 
              lam * l1(W(xn)) + 
              dot(grad_f, xn - yn) + 
              0.5 / beta * norm_sq(yn - xn) + 
              (2 * beta - gamma_n) / (2 * beta * gamma_n) * 
              norm_sq(xn - xn_prev - beta / (2 * beta - gamma_n) * dx2_n))
        losses[2] += index * tf. reshape(Wn,[]) 
        return i + 1, xn, xn_prev, grad_f, dx1_n, dx2_n, losses

    i, xn, xn_prev, grad_f, dx1, dx2, losses = \
        tf.while_loop(cond, body, [i, xn, xn_prev, grad_f, dx1, dx2, losses], swap_memory=True)
        
    loss_f = l2(T(xn), y)
    loss_g = lam * l1(W(xn))
    final_loss = loss_f + loss_g
    return [xn, final_loss, losses]

################################
# Baseline optimization schemes
################################

def steep_desc_opt(x, y, T, T_adj, W, W_adj, lam, n_iter):
    losses = [tf.zeros((n_iter), tf.float32)] * 2
    h_norm = 8.0 / Huber_param
    
    xn = x
    i = tf.constant(0, dtype=tf.int32)
    def cond(i, xn, losses):
        return i < n_iter

    def body(i, xn, losses):
        grad_f = 2 * T_adj(T(xn) - y)
        grad_h = lam * W_adj(huber_grad(W(xn), Huber_param))
        xn -= 1./(2 + lam * h_norm) * (grad_f + grad_h)
        #evaluaete losses in each iteration
        index = tf.one_hot(i,n_iter)
        losses[0] += index * l2(T(xn), y)
        losses[1] += index * lam * huber(W(xn), Huber_param)
        return i + 1, xn, losses

    i, xn, losses = \
        tf.while_loop(cond, body, [i, xn, losses], swap_memory=True)

    loss_f = l2(T(xn), y)
    loss_g = lam * huber(W(xn), Huber_param)
    final_loss = loss_f + loss_g
    return [xn, final_loss, losses]

def nesterov_opt(x, y, T, T_adj, W, W_adj, lam, n_iter):
    losses = [tf.zeros((n_iter), tf.float32)] * 2
    h_norm = 8.0 / Huber_param
    
    xn = x
    yn = x
    tn = tf.constant(1, dtype=tf.float32)
    i = tf.constant(0, dtype=tf.int32)
    def cond(i, xn, yn, tn, losses):
        return i < n_iter

    def body(i, xn, yn, tn, losses):
        xn_prev = xn
        grad_f = 2 * T_adj(T(yn)-y)
        grad_h = lam * W_adj(huber_grad(W(yn), Huber_param))
        xn = yn - 1.0 / (2 + lam * h_norm) * (grad_f + grad_h)
        tn_prev = tn
        tn = (1 + tf.sqrt(1 + 4 * tn**2)) / 2
        yn = xn + (tn_prev - 1) / tn * (xn - xn_prev)
        #evaluaete losses in each iteration
        index = tf.one_hot(i,n_iter)
        losses[0] += index * l2(T(xn), y)
        losses[1] += index * lam * huber(W(xn), Huber_param)
        return i + 1, xn, yn, tn, losses

    i, xn, yn, tn, losses = \
        tf.while_loop(cond, body, [i, xn, yn, tn, losses], swap_memory=True)

    loss_f = l2(T(xn), y)
    loss_g = lam * huber(W(xn), Huber_param)
    final_loss = loss_f + loss_g
    return [xn, final_loss, losses]

def ista_opt(x, y, T, T_adj, W, W_adj, lam, mu, n_iter):
    losses = [tf.zeros((n_iter), tf.float32)] * 2
    
    xn = x
    i = tf.constant(0, dtype=tf.int32)
    def cond(i, xn, losses):
        return i < n_iter

    def body(i, xn, losses):
        yn = xn
        grad_f = 2 * T_adj(T(yn) - y)
        xn -= gamma_n * grad_f
        xn = prox(xn, W, W_adj, gamma_n, lam, mu)
        #evaluaete losses in each iteration
        index = tf.one_hot(i,n_iter)
        losses[0] += index * l2(T(xn), y)
        losses[1] += index * lam * l1(W(xn))
        return i + 1, xn, losses

    i, xn, losses = \
        tf.while_loop(cond, body, [i, xn, losses], swap_memory=True)

    loss_f = l2(T(xn), y)
    loss_g = lam * l1(W(xn))
    final_loss = loss_f + loss_g
    return [xn, final_loss, losses]

def fista_opt(x, y, T, T_adj, W, W_adj, lam, mu, n_iter):
    losses = [tf.zeros((n_iter), tf.float32)] * 2
    
    xn = x
    yn = x
    tn = tf.constant(1, dtype=tf.float32)
    i = tf.constant(0, dtype=tf.int32)
    def cond(i, xn, yn, tn, losses):
        return i < n_iter

    def body(i, xn, yn, tn, losses):
        xn_prev = xn
        grad_f = 2 * T_adj(T(yn) - y)
        xn = yn - gamma_n * grad_f
        xn = prox(xn, W, W_adj, gamma_n, lam, mu)
        tn_prev = tn
        tn = (1 + tf.sqrt(1 + 4 * tn**2)) / 2
        yn = xn + (tn_prev - 1) / tn * (xn - xn_prev)
        #evaluaete losses in each iteration
        index = tf.one_hot(i,n_iter)
        losses[0] += index * l2(T(xn), y)
        losses[1] += index * lam * l1(W(xn))
        return i + 1, xn, yn, tn, losses

    i, xn, yn, tn, losses = \
        tf.while_loop(cond, body, [i, xn, yn, tn, losses], swap_memory=True)

    loss_f = l2(T(xn), y)
    loss_g = lam * l1(W(xn))
    final_loss = loss_f + loss_g
    return [xn, final_loss, losses]


def optimize(algorithm, x, y, T_odl, W_odl, n_iter=10):
    mu = W_odl.norm(estimate=True)**2
    # create tensorflow layers
    T = as_tensorflow_layer(T_odl)
    T_adj = as_tensorflow_layer(T_odl.adjoint)
    W = as_tensorflow_layer(W_odl)
    W_adj = as_tensorflow_layer(W_odl.adjoint)
    if algorithm == "learned_smooth":
        return learned_opt_smooth(x, y, 
                                  T, T_adj, W, W_adj,
                                  lam_smooth, n_iter)
    elif algorithm == "learned_nonsmooth":
        return learned_opt_nonsmooth(x, y, 
                                     T, T_adj, W, W_adj, 
                                     lam_nonsmooth, mu, n_iter)
    elif algorithm == "steep_desc":
        return steep_desc_opt(x, y, T, T_adj, W, W_adj, lam_smooth, n_iter)
    elif algorithm == "nesterov":
        return nesterov_opt(x, y, T, T_adj, W, W_adj, lam_smooth, n_iter)
    elif algorithm == "ista":
        return ista_opt(x, y, T, T_adj, W, W_adj, lam_nonsmooth, mu, n_iter)
    elif algorithm == "fista":
        return fista_opt(x, y, T, T_adj, W, W_adj, lam_nonsmooth, mu, n_iter)
    else:
        raise ValueError('Unknown algorithm')

