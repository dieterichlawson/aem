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
               num_x_samples=10,
               data_mean=None):
    if data_mean is None:
      self.data_mean = tf.zeros([data_dim], dtype=tf.float32)
    else:
      self.data_mean = data_mean
    self.num_x_samples = num_x_samples
    self.data_dim = data_dim
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
    # [batch_size]
    loss = tf.reduce_sum(hess + 0.5*tf.square(score), axis=-1)
    return tf.reduce_mean(loss)


 
  def sample(self, num_samples=1):
    samples = []
    for i in range(num_samples): 
      samples.append(self._sample2())
    return tf.stack(samples)

  def _sample2(self):

    sample_ta0 = tf.TensorArray(self.data_mean.dtype,
                             size=self.data_dim,
                             dynamic_size=False,
                             element_shape=[1],
                             clear_after_read=False,
                             name="samples_ta")

    def sample_body(i, sample_ta):
      # [1, i]
      prev_pixels = tf.transpose(sample_ta.gather(tf.range(0,i)))
      # [1, data_dim]
      arnn_input = tf.pad(prev_pixels, [[0, 0], [0, self.data_dim-i]], "CONSTANT")
      arnn_input = arnn_input - self.data_mean[tf.newaxis,:]
      # [1, data_dim, context_dim]
      contexts = self.arnn_net(arnn_input)
      # [1, context_dim]
      context = contexts[:,i,:]
      # [num_x_samples, context_dim] 
      tiled_context = tf.tile(context, [self.num_x_samples, 1])
      # [num_x_samples, 1]
      xs = tf.range(0, 1.0, 1./self.num_x_samples)[:,tf.newaxis]
      centered_xs = xs - self.data_mean[i]
      # [num_x_samples, context_dim+1]
      enn_input = tf.concat([centered_xs, tiled_context], axis=-1)
      # [num_x_samples]
      log_energies = self.enn_net(enn_input)
      # [1]
      log_Z_hat = (tf.math.reduce_logsumexp(log_energies, axis=0) -
        tf.log(tf.to_float(self.num_x_samples)))
      # [num_x_samples]
      weights = log_energies - log_Z_hat
      # [1]
      ind = tfd.Categorical(probs=weights).sample()
      new_sample = tf.reshape(tf.gather(tf.squeeze(xs), ind), [1])
      sample_ta = sample_ta.write(i, new_sample)
      return i+1, sample_ta

    def sample_predicate(i, unused_sample_ta):
      return tf.less(i, self.data_dim)

    results = tf.while_loop(sample_predicate,
                            sample_body, 
                            (0, sample_ta0), back_prop=False)
    
    sample = tf.reshape(results[1].stack(), [self.data_dim])
    return sample


  def _sample(self):

    sample_ta0 = tf.TensorArray(self.data_mean.dtype,
                             size=self.data_dim,
                             dynamic_size=False,
                             element_shape=[1],
                             clear_after_read=False,
                             name="samples_ta")

    def sample_body(i, sample_ta):
      # [1, i]
      prev_pixels = tf.transpose(sample_ta.gather(tf.range(0,i)))
      # [1, data_dim]
      arnn_input = tf.pad(prev_pixels, [[0, 0], [0, self.data_dim-i]], "CONSTANT")
      arnn_input = arnn_input - self.data_mean[tf.newaxis,:]
      # [1, data_dim, context_dim]
      contexts = self.arnn_net(arnn_input)
      # [1, context_dim]
      context = contexts[:,i,:]

      def le(x):
        centered_x = x - self.data_mean[i]
        # [1, context_dim+1]
        enn_input = tf.concat([centered_x, context], axis=-1)
        # [1]
        energy = self.enn_net(enn_input)
        return - energy

      # [1, 1]
      new_samples = utils.langevin(le, tf.reshape(self.data_mean[i], [1]), 
                                   step_size=0.1, num_samples=1,
                                   burn_in=100, thinning_steps=100)
      new_samples = tf.reshape(new_samples, [1])
      sample_ta = sample_ta.write(i, new_samples)
      return i+1, sample_ta

    def sample_predicate(i, unused_sample_ta):
      return tf.less(i, self.data_dim)

    results = tf.while_loop(sample_predicate,
                            sample_body, 
                            (0, sample_ta0), back_prop=False)
    
    sample = tf.reshape(results[1].stack(), [self.data_dim])
    return sample
