import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from . import base
from . import utils

class ScoreResnetSSM(object):
  """A model trained with SSM that models the score directly using a ResNet."""
  
  def __init__(self,
               data_dim,
               num_hidden_units, 
               num_res_blocks,
               activation=tf.nn.relu,
               data_mean=None,
               num_v=1):
    self.num_v = num_v
    if data_mean is None:
      self.data_mean = tf.zeros([data_dim], dtype=tf.float32)
    else:
      self.data_mean = data_mean
    with tf.variable_scope("score_resnet_ssm"):
      self.net = base.ResNet(data_dim,
                             num_hidden_units,
                             num_res_blocks,
                             activation=activation,
                             output_dim=data_dim)
  
  def grad_log_energy(self, x, summarize=True):
    return self.net(x - self.data_mean[tf.newaxis,:])

  def loss(self, x, summarize=True):
    grad_log_energy = self.grad_log_energy(x, summarize=summarize)
    tf.summary.scalar("grad log energy", tf.reduce_sum(grad_log_energy))
    loss = tf.reduce_mean(utils.ssm_with_score(grad_log_energy, x, num_v=self.num_v))
    return loss

class EnergyResnetSSM(object):
  """A model trained with SSM that models the log energy using a ResNet."""
  
  def __init__(self,
               data_dim,
               num_hidden_units, 
               num_res_blocks, 
               activation=tf.nn.relu,
               data_mean=None,
               num_v=1):
    if data_mean is None:
      self.data_mean = tf.zeros([data_dim], dtype=tf.float32)
    else:
      self.data_mean = data_mean

    self.num_v = num_v
    with tf.variable_scope("energy_resnet_ssm"):
      self.net = base.ResNet(data_dim,
                             num_hidden_units,
                             num_res_blocks,
                             activation=activation,
                             final_activation=None,#lambda x: -tf.nn.softplus(x),
                             output_dim=1)
 
  def log_energy(self, x, summarize=True):
    batch_size, data_dim = x.get_shape().as_list()
    return tf.reshape(self.net(x - self.data_mean[tf.newaxis,:]), [batch_size])

  def loss(self, x, summarize=True):
    log_energy = self.log_energy(x, summarize=summarize)
    if summarize:
      tf.summary.scalar("log_energy", tf.reduce_sum(log_energy))
    loss = tf.reduce_mean(utils.ssm(log_energy, x, num_v=self.num_v))
    return loss

  def sample(self, num_samples=1):
    energy_fn = lambda x: -self.log_energy(x, summarize=False)
    #samples = utils.hmc(energy_fn, self.data_mean, step_size=0.016, thinning_steps=100, num_samples=num_samples)
    samples = utils.langevin(energy_fn, self.data_mean, step_size=0.1,
                             thinning_steps=1000, burn_in=1000, num_samples=num_samples)
    return samples

class GaussianSSM(object):
  
  def __init__(self,
               data_dim,
               num_v=1,
               min_scale=1e-3):
    self.num_v = num_v
    with tf.variable_scope("gaussian_ssm"):
      self.mean = tf.get_variable(name="mean", shape=[data_dim],
              initializer=tf.constant_initializer(1.))
      self.scale_diag_raw = tf.get_variable(name="raw_scale_diag", shape=[data_dim],
              initializer=tf.constant_initializer(-1.351))
    scale = tf.math.softplus(self.scale_diag_raw) + min_scale
    self.dist = tfd.MultivariateNormalDiag(loc=self.mean, scale_diag=scale)
 
  def log_energy(self, x, summarize=True):
    if summarize:
      tf.summary.scalar("gaussian_params/mean_1", self.mean[0])
      tf.summary.scalar("gaussian_params/mean_2", self.mean[1])
      tf.summary.scalar("gaussian_params/scale_1", self.dist.stddev()[0])
      tf.summary.scalar("gaussian_params/scale_2", self.dist.stddev()[1])
    return self.dist.log_prob(x) 

  def loss(self, x, summarize=True):
    log_energy = self.log_energy(x, summarize=summarize)
    if summarize:
      tf.summary.scalar("log_energy", tf.reduce_sum(log_energy))
    loss = tf.reduce_mean(utils.ssm(log_energy, x, num_v=self.num_v))
    return loss
