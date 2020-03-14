import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import base

class BernoulliARM(object):
  
  def __init__(self,
               data_dim,
               arnn_num_hidden_units, arnn_num_res_blocks, context_dim,
               enn_num_hidden_units, enn_num_res_blocks,
               data_mean=None,
               enn_activation=tf.nn.relu,
               arnn_activation=tf.nn.relu):
    if data_mean is None:
      self.data_mean = tf.zeros([data_dim], dtype=tf.float32)
    else:
      self.data_mean = data_mean
    self.context_dim = context_dim
    self.data_dim = data_dim
    with tf.variable_scope("aem"):
      self.arnn_net = base.ResMADE(data_dim,
                              arnn_num_hidden_units,
                              context_dim,
                              arnn_num_res_blocks,
                              activation=arnn_activation,
                              name="arnn")
      self.enn_net = base.ResNet(
                         context_dim,
                         enn_num_hidden_units,
                         enn_num_res_blocks,
                         output_dim=1,
                         activation=enn_activation,
                         final_activation=None,
                         name="enn")
  
  def log_p(self, x, summarize=True):
    batch_size, data_dim = x.get_shape().as_list()
    # [batch_size, data_dim, context_dim]
    contexts = self.arnn_net(x - self.data_mean)
    # [batch_size, data_dim]
    logits = tf.reshape(self.enn_net(contexts), [batch_size, data_dim])
    p_dist = tfd.Bernoulli(logits=logits)
    # [batch_size, data_dim]
    log_p = p_dist.log_prob(x)
    # [batch_size]
    log_p = tf.reduce_sum(log_p, axis=1)
    if summarize:
      tf.summary.scalar("log_p", tf.reduce_mean(log_p))
    return log_p

  def loss(self, x, summarize=True):
    log_p = self.log_p(x, summarize=summarize)
    return -tf.reduce_mean(log_p)

  def sample(self, num_samples=1):
    sample_ta = tf.TensorArray(
            dtype=tf.float32, 
            size=self.data_dim, 
            dynamic_size=False,
            clear_after_read=False, 
            element_shape=[num_samples]).unstack(tf.zeros([self.data_dim, num_samples]))

    def while_body(i, sample_ta):
      contexts = self.arnn_net(tf.transpose(sample_ta.stack()) - self.data_mean)
      # [num_samples, context_dim]
      context_i = contexts[:,i,:]
      logits_i = tf.reshape(self.enn_net(context_i), [num_samples])
      p = tfd.Bernoulli(logits=logits_i)
      sample = tf.to_float(p.sample())
      sample_ta = sample_ta.write(i, sample)
      return i+1, sample_ta

    def while_cond(i, unused_ta):
      return i < self.data_dim

    outs = tf.while_loop(while_cond, while_body, (0, sample_ta), back_prop=False)
    return tf.transpose(outs[1].stack())
