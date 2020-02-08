import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def hess_diag(y, x):
  hess = tf.hessians(y, x)[0]
  hess_diag = tf.linalg.tensor_diag_part(hess)
  return hess_diag

def hess_trace(y, x):
  return tf.reduce_sum(hess_diag(y, x), axis=-1)

def ssm_with_score(score, x, num_v=1):
  batch_size, data_dim = x.get_shape().as_list()
  v_dist = tfd.Normal(loc=tf.zeros([data_dim]), 
                      scale=tf.ones([data_dim]))
  # [num_v, data_dim]
  v = v_dist.sample(num_v)
  # [batch_size, num_v]
  vt_score = tf.linalg.matmul(score, v, transpose_b=True)
  # [num_v, batch_size, data_dim]
  est1 = [tf.gradients(vt_score[:,i], x)[0] for i in range(num_v)]
  # [num_v, batch_size]
  est2 = [tf.linalg.matvec(est1[i], v[i,:]) for i in range(num_v)]
  est = tf.reduce_mean(est2, axis=0)
  return est + 0.5*tf.reduce_sum(tf.square(score), axis=1)

def ssm(y, x, num_v=1):
  # [batch_size, data_dim]
  score = tf.gradients(y, x)[0]
  return ssm_with_score(score, x, num_v=num_v)

def est_hess_trace_while(y, x, num_v=1):
  batch_size, data_dim = x.get_shape().as_list()
  v_dist = tfd.Normal(loc=tf.zeros([data_dim]), scale=tf.ones([data_dim]))
  # [num_v, data_dim]
  v = v_dist.sample(num_v)
  # [batch_size, data_dim]
  score = tf.gradients(y, x)[0]
  # [batch_size, num_v]
  vt_score = tf.linalg.matmul(score, v, transpose_b=True)

  def while_body(n, acc):
    vt_score_i = vt_score[:,n]
    v_i = v[n,:]
    grad = tf.gradients(vt_score_i, x)[0]
    diag_comp = tf.linalg.matvec(grad, v_i)
    return n+1, acc + diag_comp / num_v

  def while_predicate(n, unused_acc):
    return n < num_v

  final_vals = tf.while_loop(
          while_predicate, 
          while_body, 
          (0, tf.zeros([batch_size], dtype=tf.float32)), 
          parallel_iterations=num_v)
  return final_vals[1]

def est_hess_trace_while2(y, x, num_v=1):
  batch_size, data_dim = x.get_shape().as_list()
  v_dist = tfd.Normal(loc=tf.zeros([data_dim]), 
                      scale=tf.ones([data_dim]))
  # [num_v, data_dim]
  v = v_dist.sample(num_v)
  # [batch_size, data_dim]
  score = tf.gradients(y, x)[0]
  # [batch_size, num_v]
  vt_score = tf.linalg.matmul(score, v, transpose_b=True)
  
  grad_ta = tf.TensorArray(tf.float32, 
                           size=num_v, 
                           dynamic_size=False, 
                           element_shape=x.shape)
  
  def while_body(n, grad_ta):
    vt_score_i = vt_score[:,n]
    grad = tf.gradients(vt_score_i, x)[0]
    grad_ta = grad_ta.write(n, grad)
    return n+1, grad_ta
  
  def while_predicate(n, unused_grad_ta):
    return n < num_v

  while_outs = tf.while_loop(while_predicate, 
                             while_body, 
                             (0, grad_ta), 
                             parallel_iterations=num_v)
  # [num_v, batch_size, data_dim]
  grads = while_outs[1].stack()
  estimate = tf.linalg.matmul(grads, v[:,:,tf.newaxis])
  estimate = tf.reduce_mean(estimate, axis=0)
  return estimate
