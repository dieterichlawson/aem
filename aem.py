import os
import numpy as np
from matplotlib import cm
import functools

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import dists
import datasets
import models.aem as aem
import models.eim as eim
import models.aem_ssm as aem_ssm
import models.aem_arsm as aem_arsm
import models.resnet_ssm as resnet_ssm
import models.base as base

TARGETS = dists.TARGET_DISTS + ["dynamic_mnist", "raw_mnist", "jittered_mnist", "power", "gas"]

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_enum("target", dists.NINE_GAUSSIANS_DIST,  TARGETS,
                         "Data to train on.")
tf.app.flags.DEFINE_enum("split", "train", ["train", "valid", "test"], "Split to use.")
tf.app.flags.DEFINE_enum("model", "aem",  
                          ["aem", "eim", "aem_ssm", "energy_resnet_ssm", "score_resnet_ssm",
                           "aem_arsm", "gaussian_ssm"],
                         "Model to train.")
tf.app.flags.DEFINE_enum("activation", "tanh", ["relu", "tanh", "sigmoid"],
                         "Activation function to use for the networks.")
tf.app.flags.DEFINE_enum("enn_activation", None, ["relu", "tanh", "sigmoid"],
                         "Activation function to use for the enn network.")
tf.app.flags.DEFINE_enum("arnn_activation", None, ["relu", "tanh", "sigmoid"],
                         "Activation function to use for the arnn network.")
tf.app.flags.DEFINE_integer("arnn_num_hidden_units", 256,
                             "Number of hidden units per layer on the ARNN.")
tf.app.flags.DEFINE_integer("arnn_num_res_blocks", 4,
                             "Number of residual blocks in the ARNN.")
tf.app.flags.DEFINE_integer("context_dim", 64,
                             "Dimensionality of the context vector.")
tf.app.flags.DEFINE_integer("q_num_mixture_components", 10,
                             "Number of mixture components in q..")
tf.app.flags.DEFINE_integer("enn_num_hidden_units", 128,
                             "Number of hidden units per layer in the ENN.")
tf.app.flags.DEFINE_integer("enn_num_res_blocks", 4,
                             "Number of residual blocks in the ENN.")
tf.app.flags.DEFINE_integer("num_importance_samples", 20,
                             "Number of importance samples used to estimate Z_hat.")
tf.app.flags.DEFINE_float("learning_rate", 1e-4,
                           "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_integer("warmup_steps", 0,
                           "Number of steps to train proposal with maximum likelihood for.")
tf.app.flags.DEFINE_integer("batch_size", 256,
                             "The number of examples per batch.")
tf.app.flags.DEFINE_integer("density_num_bins", 50,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_string("tag", "aem",
                            "Name for the run.")
tf.app.flags.DEFINE_string("logdir", "/tmp/aem",
                            "Directory for summaries and checkpoints.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", 100,
                            "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS

tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])

ACTIVATION_DICT = { 
        "relu": tf.nn.relu,
        "tanh": tf.math.tanh,
        "sigmoid": tf.math.sigmoid
        }

def make_slug():
  d =[("model", FLAGS.model),
      ("target", FLAGS.target),
      ("lr", FLAGS.learning_rate),
      ("bs", FLAGS.batch_size)]
  if FLAGS.model in ["aem", "eim", "aem_ssm", "aem_arsm"]:
    d.extend([
        ("cdim", FLAGS.context_dim),
        ("arnn_act", FLAGS.arnn_activation),
        ("arnn_units", FLAGS.arnn_num_hidden_units),
        ("arnn_blocks", FLAGS.arnn_num_res_blocks),
        ("enn_act", FLAGS.enn_activation),
        ("enn_units", FLAGS.enn_num_hidden_units),
        ("enn_blocks", FLAGS.enn_num_res_blocks)])
  elif FLAGS.model == "energy_resnet_ssm":
    d.extend([
        ("enn_act", FLAGS.enn_activation),
        ("enn_units", FLAGS.enn_num_hidden_units),
        ("enn_blocks", FLAGS.enn_num_res_blocks)])
  if FLAGS.model in ["aem", "eim"]:
    d.append(("num_samples", FLAGS.num_importance_samples))
  if FLAGS.model == "aem":
    d.extend([
        ("q_num_comps", FLAGS.q_num_mixture_components),
        ("wmup_steps", FLAGS.warmup_steps)]) 
  return ".".join([FLAGS.tag] + ["%s_%s" % (k,v) for k,v in d])

def make_log_hooks(global_step, loss, logdir):
  hooks = []

  def summ_formatter(d):
    return ("Step {step}, loss: {loss:.5f}".format(**d))
  loss_hook = tf.train.LoggingTensorHook(
      {"step": global_step, "loss": loss},
      every_n_iter=FLAGS.summarize_every,
      formatter=summ_formatter)
  hooks.append(loss_hook)
  if len(tf.get_collection("infrequent_summaries")) > 0:
    infrequent_summary_hook = tf.train.SummarySaverHook(
        save_steps=FLAGS.summarize_every*3,
        output_dir=logdir,
        summary_op=tf.summary.merge_all(key="infrequent_summaries")
    )
    hooks.append(infrequent_summary_hook)
  return hooks

def make_density_image_summary(num_pts, bounds, model):
  x = tf.range(bounds[0], bounds[1], delta=(bounds[1]-bounds[0])/float(num_pts))
  X, Y = tf.meshgrid(x, x)
  XY = tf.reshape(tf.stack([X,Y], axis=-1), [num_pts**2, 2])
  tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])
  if FLAGS.model == "aem":
    log_p_hat, log_q = model.log_p(XY, num_importance_samples=FLAGS.num_importance_samples, summarize=False)
    density_p = tf.reshape(tf.exp(log_p_hat), [num_pts, num_pts])
    density_p = (density_p - tf.reduce_min(density_p))/(tf.reduce_max(density_p) -
            tf.reduce_min(density_p))
    density_q = tf.reshape(tf.exp(log_q), [num_pts, num_pts])
    density_q = (density_q - tf.reduce_min(density_q))/(tf.reduce_max(density_q) -
            tf.reduce_min(density_q))

    density_p_plot = tf_viridis(density_p)
    density_q_plot = tf_viridis(density_q)
    tf.summary.image("p_density", density_p_plot, max_outputs=1, 
            collections=["infrequent_summaries"])
    tf.summary.image("q_density", density_q_plot, max_outputs=1, 
            collections=["infrequent_summaries"])
  elif FLAGS.model == "eim":
    log_p_hat = model.log_p(XY, num_importance_samples=FLAGS.num_importance_samples, summarize=False)
    density_p = tf.reshape(tf.exp(log_p_hat), [num_pts, num_pts])
    density_p = (density_p - tf.reduce_min(density_p))/(tf.reduce_max(density_p) -
            tf.reduce_min(density_p))
    density_p_plot = tf_viridis(density_p)
    tf.summary.image("p_density", density_p_plot, max_outputs=1, 
            collections=["infrequent_summaries"])
    _, q_dist = model.arnn(XY)
    density_q = tf.reshape(tf.reduce_sum(q_dist.prob(XY), axis=-1), [num_pts, num_pts])
    density_q = (density_q - tf.reduce_min(density_q))/(tf.reduce_max(density_q) -
            tf.reduce_min(density_q))
    density_q_plot = tf_viridis(density_q)
    tf.summary.image("q_density", density_q_plot, max_outputs=1, 
            collections=["infrequent_summaries"])
  elif (FLAGS.model == "energy_resnet_ssm" or 
        FLAGS.model == "aem_ssm" or
        FLAGS.model == "aem_arsm" or
        FLAGS.model == "gaussian_ssm"):
    log_energy = model.log_energy(XY, summarize=False)
    density_p = tf.reshape(tf.exp(log_energy), [num_pts, num_pts])
    density_p = (density_p - tf.reduce_min(density_p))/(tf.reduce_max(density_p) -
            tf.reduce_min(density_p))
    density_p_plot = tf_viridis(density_p)
    tf.summary.image("p_density", density_p_plot, max_outputs=1, 
            collections=["infrequent_summaries"])


def main(unused_argv):
  with tf.device('/GPU:0'):
    g = tf.Graph()
    with g.as_default():

      if FLAGS.target in dists.TARGET_DISTS:
        data = dists.get_target_distribution(FLAGS.target).sample(FLAGS.batch_size)
        _, data_dim = data.get_shape().as_list()
        mean = None
      else:
        data, mean, _ = datasets.get_dataset(
                FLAGS.target, FLAGS.batch_size, FLAGS.split, 
                shuffle=True, repeat=True, initializable=False)
        data = tf.reshape(data, [FLAGS.batch_size, -1])
        mean = tf.reshape(mean, [-1])
        _, data_dim = data.get_shape().as_list()
    
      if FLAGS.enn_activation is None:
        FLAGS.enn_activation = FLAGS.activation
      if FLAGS.arnn_activation is None:
        FLAGS.arnn_activation = FLAGS.activation

      enn_activation = ACTIVATION_DICT[FLAGS.enn_activation]
      arnn_activation = ACTIVATION_DICT[FLAGS.arnn_activation]
      
      if FLAGS.model == "eim" and FLAGS.target == "jittered_mnist":
        squash=True
      else:
        squash=False

      global_step = tf.train.get_or_create_global_step()
      warmup_step_ind = tf.cast(global_step > FLAGS.warmup_steps, tf.float32)

      if FLAGS.model == "aem":
        model = aem.AEM(data_dim,
                        arnn_num_hidden_units=FLAGS.arnn_num_hidden_units, 
                        arnn_num_res_blocks=FLAGS.arnn_num_res_blocks, 
                        context_dim=FLAGS.context_dim, 
                        enn_num_hidden_units=FLAGS.enn_num_hidden_units, 
                        enn_num_res_blocks=FLAGS.enn_num_res_blocks, 
                        num_importance_samples=FLAGS.num_importance_samples,
                        q_num_mixture_comps=FLAGS.q_num_mixture_components, 
                        enn_activation=enn_activation,
                        arnn_activation=arnn_activation,
                        data_mean=mean,
                        warmup_step_ind=warmup_step_ind,
                        q_min_scale=1e-3)
      elif FLAGS.model == "eim":
        model = eim.EIM(data_dim,
                        arnn_num_hidden_units=FLAGS.arnn_num_hidden_units, 
                        arnn_num_res_blocks=FLAGS.arnn_num_res_blocks, 
                        context_dim=FLAGS.context_dim, 
                        enn_num_hidden_units=FLAGS.enn_num_hidden_units, 
                        enn_num_res_blocks=FLAGS.enn_num_res_blocks, 
                        num_importance_samples=FLAGS.num_importance_samples,
                        proposal_activation=arnn_activation,
                        energy_activation=enn_activation,
                        data_mean=None if squash else mean, 
                        q_min_scale=1e-3)
        model = base.SquashedDistribution(model, mean)
      elif FLAGS.model == "aem_ssm":
        model = aem_ssm.AEMSSM(data_dim,
                        arnn_num_hidden_units=FLAGS.arnn_num_hidden_units, 
                        arnn_num_res_blocks=FLAGS.arnn_num_res_blocks, 
                        context_dim=FLAGS.context_dim, 
                        enn_num_hidden_units=FLAGS.enn_num_hidden_units, 
                        enn_num_res_blocks=FLAGS.enn_num_res_blocks,
                        data_mean=mean,
                        arnn_activation=arnn_activation,
                        enn_activation=enn_activation)
      elif FLAGS.model == "aem_arsm":
        model = aem_arsm.AEMARSM(data_dim,
                        arnn_num_hidden_units=FLAGS.arnn_num_hidden_units, 
                        arnn_num_res_blocks=FLAGS.arnn_num_res_blocks, 
                        context_dim=FLAGS.context_dim, 
                        enn_num_hidden_units=FLAGS.enn_num_hidden_units, 
                        enn_num_res_blocks=FLAGS.enn_num_res_blocks,
                        data_mean=mean,
                        arnn_activation=arnn_activation,
                        enn_activation=enn_activation)
      elif FLAGS.model == "score_resnet_ssm":
        model = resnet_ssm.ScoreResnetSSM(
                data_dim,
                num_hidden_units=FLAGS.enn_num_hidden_units, 
                num_res_blocks=FLAGS.enn_num_res_blocks, 
                activation=enn_activation,
                data_mean=mean,
                num_v=1)
      elif FLAGS.model == "energy_resnet_ssm":
        model = resnet_ssm.EnergyResnetSSM(
                data_dim,
                num_hidden_units=FLAGS.enn_num_hidden_units, 
                num_res_blocks=FLAGS.enn_num_res_blocks, 
                activation=enn_activation,
                data_mean=mean,
                num_v=1)
      elif FLAGS.model == "gaussian_ssm":
        model = resnet_ssm.GaussianSSM(data_dim)

      loss = model.loss(data, summarize=True)
      tf.summary.scalar("loss", loss)

      if "mnist" in FLAGS.target and FLAGS.model in ["aem", "eim", "aem_ssm", "aem_arsm", "energy_resnet_ssm"]:
        sample = model.sample(num_samples=4)
        sample = tf.reshape(sample, [4, 28, 28, 1])
        tf.summary.image("sample", sample, max_outputs=4, 
              collections=["infrequent_summaries"])
      
      if FLAGS.target in dists.TARGET_DISTS:
        make_density_image_summary(FLAGS.density_num_bins, (-2,2), model)

      learning_rate = tf.train.cosine_decay(FLAGS.learning_rate, 
              global_step, FLAGS.max_steps, name=None)
      tf.summary.scalar("learning_rate", learning_rate)
      opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = opt.minimize(loss, global_step=global_step)
      logdir = os.path.join(FLAGS.logdir, make_slug())
      log_hooks = make_log_hooks(global_step, loss, logdir)
      with tf.train.MonitoredTrainingSession(
          master="",
          is_chief=True,
          hooks=log_hooks,
          checkpoint_dir=logdir,
          save_checkpoint_secs=60,
          save_summaries_steps=FLAGS.summarize_every,
          log_step_count_steps=FLAGS.summarize_every) as sess:
        cur_step = -1
        while True:
          if sess.should_stop() or cur_step > FLAGS.max_steps:
            break
          # run a step
          _, cur_step = sess.run([train_op, global_step])

if __name__ == "__main__":
  tf.app.run(main)
