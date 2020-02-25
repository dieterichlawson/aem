import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import base
from . import utils

class AEMARSM(object):
  
  def __init__(self,
               data_dim,
               arnn_num_hidden_units, 
               arnn_num_res_blocks, 
               context_dim,
               enn_num_hidden_units, 
               enn_num_res_blocks, 
               arnn_activation=tf.nn.relu,
               enn_activation=tf.nn.relu,
               data_mean=None):
    if data_mean is None:
      self.data_mean = tf.zeros([data_dim], dtype=tf.float32)
    else:
      self.data_mean = data_mean
    self.context_dim = context_dim
    with tf.variable_scope("aem"):
      num_outputs_per_dim = context_dim
      self.arnn_net = base.ResMADE(data_dim,
                                   arnn_num_hidden_units,
                                   context_dim,
                                   arnn_num_res_blocks,
                                   activation=arnn_activation,
                                   name="arnn")
      self.enn_net = base.ENN(context_dim + 1,
                              enn_num_hidden_units,
                              enn_num_res_blocks,
                              activation=enn_activation,
                              name="enn")
  
  def log_energy(self, x, summarize=True):
    batch_size, data_dim = x.get_shape().as_list()
    centered_x = x - self.data_mean[tf.newaxis,:]
    # [batch_size, data_dim, context_dim]
    contexts = self.arnn_net(centered_x)
    # [batch_size, data_dim, context_dim+1]
    enn_input = tf.concat([centered_x[:,:,tf.newaxis], contexts], axis=-1)
    log_energies = self.enn_net(enn_input)
    log_energies = tf.reshape(log_energies, [batch_size, data_dim])
    # [batch_size]
    log_energy = tf.reduce_sum(log_energies, axis=-1)
    if summarize:
      tf.summary.scalar("log_energy", tf.reduce_mean(tf.reduce_sum(log_energies, axis=-1)))
    return log_energy

  def loss(self, x, summarize=True):
    batch_size, data_dim = x.get_shape().as_list()
    centered_x = x - self.data_mean[tf.newaxis,:]
    # [batch_size, data_dim, context_dim]
    contexts = self.arnn_net(centered_x)
    # [batch_size, data_dim, context_dim+1]
    enn_input = tf.concat([centered_x[:,:,tf.newaxis], contexts], axis=-1)
    log_energies = self.enn_net(enn_input)
    log_energies = tf.reshape(log_energies, [batch_size, data_dim])
    # [batch_size]
    log_energy = tf.reduce_sum(log_energies, axis=-1)
    if summarize:
      tf.summary.scalar("log_energy", tf.reduce_mean(tf.reduce_sum(log_energies, axis=-1)))
    
    # [batch_size, data_dim]
    score = tf.gradients(log_energy, x, stop_gradients=[contexts])[0]
    # [batch_size, data_dim]
    hess = tf.gradients(score, x, stop_gradients=[contexts])[0]
    batch_size
    loss = tf.reduce_sum(hess + 0.5*tf.square(score), axis=-1)
    return tf.reduce_mean(loss)