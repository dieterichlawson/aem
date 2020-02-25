import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow_datasets as tfds
import numpy as np
import os

flags = tf.flags

flags.DEFINE_string("data_dir", None, "Directory to store datasets.")
FLAGS = flags.FLAGS

def get_uci_power_dataset(split, shuffle_files=False):
  return get_uci_dataset("power", split=split)

def get_uci_gas_dataset(split, shuffle_files=False):
  return get_uci_dataset("gas", split=split)


def get_uci_dataset(dataset_name, split="train"):
  path = os.path.join(FLAGS.data_dir, "data","processed", dataset_name, "%s.npy" % split)
  data = np.load(path)
  num_pts, data_dim = data.shape
  dataset = tf.data.Dataset.from_tensor_slices(data)
  return dataset, tf.zeros([data_dim], dtype=tf.float32)


def get_static_mnist(split="train", shuffle_files=False):
  """Get Static Binarized MNIST dataset."""
  split_map = {
      "train": "train",
      "valid": "validation",
      "test": "test",
  }
  preprocess = lambda x: tf.cast(x["image"], tf.float32) * 255.
  datasets = tfds.load(name="binarized_mnist",
                       shuffle_files=shuffle_files,
                       data_dir=FLAGS.data_dir)
  train_mean = compute_mean(datasets[split_map["train"]].map(preprocess))
  return datasets[split_map[split]].map(preprocess), train_mean

def get_mnist(split="train", shuffle_files=False):
  """Get MNIST dataset."""
  split_map = {
      "train": "train",
      "valid": "validation",
      "test": "test",
  }
  datasets = dict(
      zip(["train", "validation", "test"],
          tfds.load(
              "mnist:3.*.*",
              split=["train[:50000]", "train[50000:]", "test"],
              shuffle_files=shuffle_files,
              data_dir=FLAGS.data_dir)))
  preprocess = lambda x: tf.to_float(x["image"])
  train_mean = compute_mean(datasets[split_map["train"]].map(preprocess))
  return datasets[split_map[split]].map(preprocess), train_mean

def compute_mean(dataset):
  def _helper(aggregate, x):
    total, n = aggregate
    return total + x, n + 1

  total, n = tfds.as_numpy(dataset.reduce((0., 0), _helper))
  return tf.to_float(total / n)

def dataset_and_mean_to_batch(dataset,
                              train_mean,
                              batch_size,
                              binarize=False,
                              repeat=True,
                              shuffle=True,
                              initializable=False,
                              preprocess=True,
                              jitter=False):
  """Transforms data based on args (assumes images in [0, 255])."""

  def jitter_im(im):
    jitter_noise = tfd.Uniform(
        low=tf.zeros_like(im), high=tf.ones_like(im)).sample()
    jittered_im = im + jitter_noise
    return jittered_im

  def _preprocess(im):
    """Preprocess the image."""
    assert not (jitter and
                binarize), "Can only choose binarize or jitter, not both."

    if jitter:
      im = jitter_im(im)
    elif binarize:
      im = tfd.Bernoulli(probs=im / 255.).sample()
    else:  # [0, 1]
      im /= 255.

    return im
  if preprocess:
    dataset = dataset.map(_preprocess)

  if repeat:
    dataset = dataset.repeat()

  if shuffle:
    dataset = dataset.shuffle(1024)

  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  if initializable:
    itr = dataset.make_initializable_iterator()
  else:
    itr = dataset.make_one_shot_iterator()

  ims = itr.get_next()

  if jitter:
    train_mean += 0.5
  elif binarize:
    train_mean /= 255.
  else:
    train_mean /= 255.

  ims_shape = ims.get_shape().as_list()
  true_shape = [batch_size] + ims_shape[1:]
  ims = tf.reshape(ims, true_shape)
  return ims, train_mean[None], itr


def get_dataset(dataset,
                batch_size,
                split,
                repeat=True,
                shuffle=True,
                initializable=False):
  """Return the reference dataset with options."""
  dataset_map = {
      "dynamic_mnist": (get_mnist, {
          "binarize": True,
      }),
      "raw_mnist": (get_mnist, {}),
      "static_mnist": (get_static_mnist, {}),
      "jittered_mnist": (get_mnist, {
          "jitter": True,
      }),
      "power": (get_uci_power_dataset, {
          "preprocess": False,
      }),
      "gas": (get_uci_gas_dataset, {
          "preprocess": False,
      }),

      #"jittered_celeba": (get_celeba, {
      #    "jitter": True
      #}),
      #"fashion_mnist": (get_fashion_mnist, {
      #    "binarize": True
      #}),
      #"jittered_fashion_mnist": (get_fashion_mnist, {
      #    "jitter": True,
      #}),
  }

  dataset_fn, dataset_kwargs = dataset_map[dataset]
  raw_dataset, mean = dataset_fn(split, shuffle_files=False)
  data_batch, mean, itr = dataset_and_mean_to_batch(
      raw_dataset,
      mean,
      batch_size=batch_size,
      repeat=repeat,
      shuffle=shuffle,
      initializable=initializable,
      **dataset_kwargs)

  return tf.cast(data_batch, tf.float32), mean, itr
