import math
import os
import numpy as np
from matplotlib import cm
import functools

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import dists
import models

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_enum("target", dists.NINE_GAUSSIANS_DIST,  dists.TARGET_DISTS,
                         "Distribution to draw data from.")
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
tf.app.flags.DEFINE_integer("batch_size", 256,
                             "The number of examples per batch.")
tf.app.flags.DEFINE_integer("density_num_bins", 50,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_integer("density_num_samples", 100000,
                            "Number of samples to use when plotting density.")
tf.app.flags.DEFINE_string("logdir", "/tmp/aem",
                            "Directory for summaries and checkpoints.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                            "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS

tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])

def make_log_hooks(global_step, loss):
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
        save_steps=1000,
        output_dir=FLAGS.logdir,
        summary_op=tf.summary.merge_all(key="infrequent_summaries")
    )
    hooks.append(infrequent_summary_hook)
  return hooks

def make_density_image_summary(num_pts, bounds, model):
  x = tf.range(bounds[0], bounds[1], delta=(bounds[1]-bounds[0])/float(num_pts))
  X, Y = tf.meshgrid(x, x)
  XY = tf.reshape(tf.stack([X,Y], axis=-1), [num_pts**2, 2])
  log_p_hat, _ = model.log_p(XY, num_importance_samples=20)
  density = tf.reshape(tf.exp(log_p_hat), [num_pts, num_pts])
  density = (density - tf.reduce_min(density))/(tf.reduce_max(density) - tf.reduce_min(density))
  tf_viridis = lambda x: tf.py_func(cm.get_cmap('viridis'), [x], [tf.float64])
  density_plot = tf_viridis(density)
  tf.summary.image("density", density_plot, max_outputs=1, collections=["infrequent_summaries"])

def main(unused_argv):
  g = tf.Graph()
  with g.as_default():

    data = dists.get_target_distribution(FLAGS.target).sample(FLAGS.batch_size)
    _, data_dim = data.get_shape().as_list()
    model = models.AEM(data_dim,
                       arnn_num_hidden_units=FLAGS.arnn_num_hidden_units, 
                       arnn_num_res_blocks=FLAGS.arnn_num_res_blocks, 
                       context_dim=FLAGS.context_dim, 
                       enn_num_hidden_units=FLAGS.enn_num_hidden_units, 
                       enn_num_res_blocks=FLAGS.enn_num_res_blocks, 
                       num_importance_samples=FLAGS.num_importance_samples,
                       q_num_mixture_comps=FLAGS.q_num_mixture_components, 
                       q_min_scale=1e-3)
    loss = model.loss(data)
    tf.summary.scalar("loss", loss)
    make_density_image_summary(FLAGS.density_num_bins, (-2,2), model)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    grads = opt.compute_gradients(-loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    log_hooks = make_log_hooks(global_step, loss) 

    with tf.train.MonitoredTrainingSession(
        master="",
        is_chief=True,
        hooks=log_hooks,
        checkpoint_dir=FLAGS.logdir,
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
