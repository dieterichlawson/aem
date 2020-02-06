import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import base

class AEM(object):
  
  def __init__(self,
               data_dim,
               arnn_num_hidden_units, arnn_num_res_blocks, context_dim,
               enn_num_hidden_units, enn_num_res_blocks,
               num_importance_samples,
               q_num_mixture_comps,
               q_min_scale=1e-3):
    self.context_dim = context_dim
    self.num_mixture_comps = q_num_mixture_comps
    self.min_scale = q_min_scale
    self.num_importance_samples = num_importance_samples
    with tf.variable_scope("aem"):
      num_outputs_per_dim = context_dim + 3*q_num_mixture_comps
      self.arnn_net = base.ResMADE(data_dim,
                              arnn_num_hidden_units,
                              num_outputs_per_dim,
                              arnn_num_res_blocks,
                              name="arnn")
      self.enn_net = base.ENN(context_dim + 1,
                         enn_num_hidden_units,
                         enn_num_res_blocks,
                         final_activation=lambda x: -tf.nn.softplus(x),
                         name="enn")
  
  def arnn(self, x):
    batch_size, data_dim = x.get_shape().as_list()
    cd = self.context_dim
    arnn_net_outs = self.arnn_net(x)
    contexts = arnn_net_outs[:,:,0:cd]
    # Construct the mixture
    mixture_weights = tf.nn.softmax(arnn_net_outs[:,:,cd:cd + self.num_mixture_comps], axis=-1)
    mixture_means = arnn_net_outs[:,:,cd + self.num_mixture_comps: cd + self.num_mixture_comps*2]
    mixture_raw_scales = arnn_net_outs[:,:,cd + self.num_mixture_comps*2:]
    mixture_scales = tf.nn.softplus(mixture_raw_scales) + self.min_scale
    if self.num_mixture_comps > 1:
      q = tfd.Mixture(
        cat=tfd.Categorical(probs=mixture_weights),
        components = [tfd.Normal(loc=mixture_means[:,:,i], scale=mixture_scales[:,:,i]) for i in range(self.num_mixture_comps)]
      )
    else:
      mixture_means = tf.reshape(mixture_means, [batch_size, data_dim])
      mixture_scales = tf.reshape(mixture_scales, [batch_size, data_dim])
      q = tfd.Normal(loc=mixture_means, scale=mixture_scales)
    return contexts, q

  def log_p(self, x, num_importance_samples=None, summarize=True):
    if num_importance_samples is None:
      nis = self.num_importance_samples
    else:
      nis = num_importance_samples
    batch_size, data_dim = x.get_shape().as_list()
    contexts, q = self.arnn(x)
    # [num_importance_samples, batch_size, data_dim]
    samples = tf.stop_gradient(q.sample(nis))
    # [num_importance_samples, batch_size, data_dim]
    sample_log_q = tf.stop_gradient(q.log_prob(samples))
    xs = tf.concat([samples, x[tf.newaxis,:,:]], axis=0) # [num_importances_samples+1, batch_size, data_dim]
    contexts = tf.tile(contexts[tf.newaxis,:,:,:], [nis+1,1,1,1]) # [num_importance_samples+1, batch_size, data_dim, context_dim]
    enn_input = tf.concat([xs[:,:,:,tf.newaxis], contexts], axis=-1)
    enn_input = tf.reshape(enn_input, [(nis+1)*batch_size, data_dim, self.context_dim+1])
    log_energies = self.enn_net(enn_input)
    log_energies = tf.reshape(log_energies, [nis+1, batch_size, data_dim])
    sample_log_energies = log_energies[0:nis,:,:] # [num_importance_samples, batch_size, data_dim]
    data_log_energies = log_energies[nis,:,:] # [batch_size, data_dim]
    # [batch_size, data_dim]
    log_Z_hat = tf.math.reduce_logsumexp(sample_log_energies - sample_log_q, axis=0) - tf.log(tf.to_float(nis))
    log_p_hat = tf.math.reduce_sum(data_log_energies - log_Z_hat, axis=-1) # [batch_size]
    log_q_data = tf.math.reduce_sum(q.log_prob(x), axis=-1) # [batch_size]
    if summarize:
      tf.summary.scalar("log_p", tf.reduce_mean(log_p_hat))
      tf.summary.scalar("log_q", tf.reduce_mean(log_q_data))
    return log_p_hat,  log_q_data

  def loss(self, x, num_importance_samples=None, summarize=True):
    log_p, log_q = self.log_p(x,
                              num_importance_samples=num_importance_samples,
                              summarize=summarize)
    return -tf.reduce_mean(log_p + log_q)

