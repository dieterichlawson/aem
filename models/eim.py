import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import base

class EIM(object):
  
  def __init__(self,
               data_dim,
               arnn_num_hidden_units, arnn_num_res_blocks, context_dim,
               enn_num_hidden_units, enn_num_res_blocks,
               num_importance_samples,
               activation=tf.nn.relu,
               q_min_scale=1e-3):
    self.context_dim = context_dim
    self.min_scale = q_min_scale
    self.num_importance_samples = num_importance_samples
    with tf.variable_scope("aem"):
      num_outputs_per_dim = context_dim + 2
      self.arnn_net = base.ResMADE(data_dim,
                                   arnn_num_hidden_units,
                                   num_outputs_per_dim,
                                   arnn_num_res_blocks,
                                   activation=activation,
                                   name="arnn")
      self.enn_net = base.ENN(context_dim + 1,
                              enn_num_hidden_units,
                              enn_num_res_blocks,
                              activation=activation,
                              name="enn")
  
  def arnn(self, x):
    batch_size, data_dim = x.get_shape().as_list()
    cd = self.context_dim
    arnn_net_outs = self.arnn_net(x)
    contexts = arnn_net_outs[:,:,0:cd]
    # Construct the mixture
    mixture_means = arnn_net_outs[:,:,cd]
    mixture_raw_scales = arnn_net_outs[:,:,cd + 1]
    mixture_scales = tf.nn.softplus(mixture_raw_scales) + self.min_scale
    proposal = tfd.Normal(loc=mixture_means, scale=mixture_scales)
    return contexts, proposal

  def log_p(self, x, num_importance_samples=None, summarize=True):
    if num_importance_samples is None:
      nis = self.num_importance_samples
    else:
      nis = num_importance_samples
    batch_size, data_dim = x.get_shape().as_list()
    contexts, proposal = self.arnn(x)
    # [num_importance_samples, batch_size, data_dim]
    samples = proposal.sample(nis)
    # [num_importance_samples, batch_size, data_dim]
    #sample_log_q = proposal.log_prob(samples)
    xs = tf.concat([samples, x[tf.newaxis,:,:]], axis=0) # [num_importances_samples+1, batch_size, data_dim]
    contexts = tf.tile(contexts[tf.newaxis,:,:,:], [nis+1,1,1,1]) # [num_importance_samples+1, batch_size, data_dim, context_dim]
    enn_input = tf.concat([xs[:,:,:,tf.newaxis], contexts], axis=-1)
    enn_input = tf.reshape(enn_input, [(nis+1)*batch_size, data_dim, self.context_dim+1])
    log_energies = self.enn_net(enn_input)
    log_energies = tf.reshape(log_energies, [nis+1, batch_size, data_dim])
    sample_log_energies = log_energies[0:nis,:,:] # [num_importance_samples, batch_size, data_dim]
    data_log_energies = tf.reduce_sum(log_energies[nis,:,:], axis=-1) # [batch_size]
    # [batch_size, data_dim]
    log_Z_hat = tf.math.reduce_logsumexp(log_energies, axis=0) - tf.log(tf.to_float(nis+1)) 
    log_Z_hat = tf.reduce_sum(log_Z_hat, axis=-1) # [batch_size]
    #log_Z_hat = tf.math.reduce_logsumexp(sample_log_energies - sample_log_q, axis=0) - tf.log(tf.to_float(nis))
    #log_p_hat = tf.math.reduce_sum(data_log_energies - log_Z_hat, axis=-1) # [batch_size]
    log_q_data = tf.math.reduce_sum(proposal.log_prob(x), axis=-1) # [batch_size]
    log_p_lower_bound = log_q_data + data_log_energies - log_Z_hat #[batch_size]
    return log_p_lower_bound

  def loss(self, x, num_importance_samples=None, summarize=True):
    return -tf.reduce_mean(self.log_p(x,
                                      num_importance_samples=num_importance_samples,
                                      summarize=summarize))
