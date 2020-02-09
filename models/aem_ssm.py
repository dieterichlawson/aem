import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import base
from . import utils

class AEMSSM(object):
  
  def __init__(self,
               data_dim,
               arnn_num_hidden_units, 
               arnn_num_res_blocks, 
               context_dim,
               enn_num_hidden_units, 
               enn_num_res_blocks, 
               activation=tf.nn.relu,
               num_v=1):
    self.context_dim = context_dim
    self.num_v = num_v
    with tf.variable_scope("aem"):
      num_outputs_per_dim = context_dim
      self.arnn_net = base.ResMADE(data_dim,
                                   arnn_num_hidden_units,
                                   context_dim,
                                   arnn_num_res_blocks,
                                   activation=activation,
                                   name="arnn")
      self.enn_net = base.ENN(context_dim + 1,
                              enn_num_hidden_units,
                              enn_num_res_blocks,
                              activation=activation,
                              name="enn")
  
  def log_energy(self, x, summarize=True):
    batch_size, data_dim = x.get_shape().as_list()
    # [batch_size, data_dim, context_dim]
    contexts = self.arnn_net(x)
    # [batch_size, data_dim, context_dim+1]
    enn_input = tf.concat([x[:,:,tf.newaxis], contexts], axis=-1)
    log_energies = self.enn_net(enn_input)
    log_energies = tf.reshape(log_energies, [batch_size, data_dim])
    # [batch_size]
    log_energy = tf.reduce_sum(log_energies, axis=-1)
    if summarize:
      tf.summary.scalar("log_energy", tf.reduce_mean(tf.reduce_sum(log_energies, axis=-1)))
    return log_energy

  def loss(self, x, summarize=True):
    batch_size, data_dim = x.get_shape().as_list()
    log_energy = self.log_energy(x, summarize=summarize)
    loss = tf.reduce_mean(utils.ssm(log_energy, x, num_v=self.num_v))
    return loss
