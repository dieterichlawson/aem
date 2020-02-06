import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class DenseLayer(object):
  
  def __init__(self, 
               num_inputs, 
               num_outputs, 
               activation=None, 
               bias_initializer=tf.zeros_initializer,
               weight_initializer=tf.glorot_uniform_initializer,
               name="dense_layer"):
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.activation = activation
    with tf.variable_scope(name):
      self.W = tf.get_variable(name="W", shape=[num_outputs, num_inputs],
              initializer=weight_initializer)
      self.b = tf.get_variable(name="b", shape=[num_outputs], 
              initializer=bias_initializer)
  
  def __call__(self, x):
    batch_size, input_dim = x.get_shape().as_list()
    assert input_dim == self.num_inputs, "%s, %s " %(input_dim, self.num_inputs)
    out = tf.linalg.matmul(x, self.W, transpose_b=True) + self.b[tf.newaxis,:]
    if self.activation is not None:
      out = self.activation(out)
    return out

class ResBlock(object):
  
  def __init__(self, 
               num_inputs, 
               num_outputs, 
               name="resmade_block"):
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    with tf.variable_scope(name):
      self.layer1 = DenseLayer(num_inputs, 
                               num_outputs, 
                               activation=tf.nn.relu, 
                               name="layer1",
                               weight_initializer=tf.variance_scaling_initializer(
                                   scale=2.0, distribution="normal"))
      self.layer2 = DenseLayer(num_inputs, 
                               num_outputs, 
                               activation=None, 
                               name="layer2",
                               weight_initializer=tf.variance_scaling_initializer(
                                   scale=0.1, distribution="normal"))
      
  def __call__(self, in_x):
    residual = tf.nn.relu(in_x)
    residual = self.layer1(residual)
    residual = self.layer2(residual)
    return in_x + residual
  

class MaskedDenseLayer(DenseLayer):
  
  def __init__(self, 
               num_inputs, 
               num_outputs, 
               data_dim, 
               activation=None, 
               mask_type="hidden", 
               bias_initializer=tf.zeros_initializer,
               weight_initializer=tf.glorot_normal_initializer,
               name="masked_layer"):
    self.data_dim = data_dim
    self.mask_type = mask_type
    
    assert data_dim > 1, "Must have data dimension > 1."
    assert mask_type in ["input", "output", "hidden"], "Mask type not 'input', 'output', or 'hidden'."
    max_degree = data_dim - 1
    if mask_type == "hidden":
      in_degrees = (tf.range(num_inputs) % max_degree) + 1
      out_degrees = (tf.range(num_outputs) % max_degree) + 1
      self.mask = tf.cast(out_degrees[:,tf.newaxis] >= in_degrees, tf.float32)
    elif mask_type == "input":
      assert num_inputs == data_dim, "For an input layer the num inputs and data dim must match."
      in_degrees = tf.range(num_inputs) + 1
      out_degrees = (tf.range(num_outputs) % max_degree) + 1
      self.mask = tf.cast(out_degrees[:,tf.newaxis] >= in_degrees, tf.float32)
    elif mask_type == "output":
      assert num_outputs % data_dim == 0, "For output layers, num_outputs must be a multiple of data_dim."
      in_degrees = (tf.range(num_inputs) % max_degree) + 1
      out_degrees = tf.tile(tf.range(data_dim) + 1, [int(num_outputs/data_dim)])
      self.mask = tf.cast(out_degrees[:,tf.newaxis] > in_degrees, tf.float32)
    super().__init__(num_inputs, num_outputs, activation=activation,
            bias_initializer=bias_initializer, weight_initializer=weight_initializer, name=name)
    
  def __call__(self, x):
    batch_size, input_dim = x.get_shape().as_list()
    assert input_dim == self.num_inputs
    out = tf.linalg.matmul(x, self.W*self.mask, transpose_b=True) + self.b[tf.newaxis,:]
    #out = tf.linalg.matmul(x, self.W, transpose_b=True) + self.b[tf.newaxis,:]
    if self.activation is not None:
      out = self.activation(out)
    return out

class MaskedResBlock(ResBlock):
  
  def __init__(self, 
               num_inputs, 
               num_outputs, 
               data_dim, 
               name="resmade_block"):
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    with tf.variable_scope(name):
      self.layer1 = MaskedDenseLayer(num_inputs,
                                     num_outputs, 
                                     data_dim, 
                                     activation=tf.nn.relu, 
                                     mask_type="hidden",
                                     weight_initializer=tf.variance_scaling_initializer(
                                         scale=2.0, distribution="normal"),
                                     name="layer1")
      self.layer2 = MaskedDenseLayer(num_inputs, 
                                     num_outputs, 
                                     data_dim, 
                                     activation=None, 
                                     mask_type="hidden", 
                                     weight_initializer=tf.variance_scaling_initializer(
                                         scale=0.1, distribution="normal"),
                                     name="layer2")
      
  
class ResMADE(object):

  def __init__(self, 
               data_dim, 
               num_hidden_units, 
               num_outputs_per_dim, 
               num_res_blocks, 
               name="resmade"):
    self.num_outputs_per_dim = num_outputs_per_dim
    with tf.variable_scope("resmade"):
      self.first_layer = MaskedDenseLayer(data_dim, 
                                     num_hidden_units, 
                                     data_dim, 
                                     activation=None, 
                                     mask_type="input", 
                                     name="first_layer")
      self.inner_layers = [
        MaskedResBlock(num_hidden_units,
                       num_hidden_units, 
                       data_dim, 
                       name="block%d" % (i+1)) 
        for i in range(num_res_blocks)
      ]
      self.final_layer = MaskedDenseLayer(num_hidden_units, 
                                          num_outputs_per_dim*data_dim, 
                                          data_dim, 
                                          activation=None, 
                                          mask_type="output", 
                                          bias_initializer=tf.glorot_normal_initializer,
                                          name="final_layer")

  def __call__(self, x):
    batch_size, data_dim = x.get_shape().as_list()
    x = self.first_layer(x)
    for layer in self.inner_layers:
      x = layer(x)
    x = tf.nn.relu(x)
    out = self.final_layer(x)
    reshaped =  tf.reshape(out, [batch_size, self.num_outputs_per_dim, data_dim])
    return tf.transpose(reshaped, [0,2,1])
  
  
class ENN(object):

  def __init__(self, 
          data_dim, 
          num_hidden_units, 
          num_res_blocks, 
          final_activation=None,
          name="enn"):
    with tf.variable_scope("enn"):
      self.first_layer = DenseLayer(data_dim,
                                    num_hidden_units,
                                    activation=None,
                                    name="first_layer")
      self.inner_layers = [
        ResBlock(num_hidden_units,
                 num_hidden_units,
                 name="block%d" % (i+1))
        for i in range(num_res_blocks)
      ]
      self.final_layer = DenseLayer(num_hidden_units,
                                    1,
                                    activation=final_activation,
                                    name="final_layer")

  def __call__(self, x):
    batch_size, data_dim, context_dim = x.get_shape().as_list()
    x = tf.reshape(x, [batch_size*data_dim, context_dim])
    x = self.first_layer(x)
    for layer in self.inner_layers:
      x = layer(x)
    x = tf.nn.relu(x)
    out = self.final_layer(x)
    return tf.reshape(out, [batch_size, data_dim])
