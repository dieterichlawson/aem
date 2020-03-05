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


def hmc(energy_fn, init_X, L=20, step_size=1.0, burn_in=100, num_samples=1000, thinning_steps=1, max_steps=None):

  samples = tf.TensorArray(init_X.dtype, 
                           size=num_samples*thinning_steps, 
                           dynamic_size=False, 
                           name='samples_ta')
  init_X = init_X[tf.newaxis,:] 
  X_shape = tf.shape(init_X)
  
  if max_steps == None:
    max_steps = 1000*num_samples*thinning_steps

  def hmc_step(i, num_accepted, q, samples):
    # Sample momentum variables as standard Gaussians.
    p = tf.random.normal(X_shape, mean=0., stddev=1.)
    init_q = q 
    # Compute initial kinetic and potential energies.
    init_K = tf.reduce_sum(tf.square(p))/2.
    init_U = energy_fn(q)
    
    # Do first half-step
    p = p - step_size * tf.gradients(init_U, q)[0] / 2.
    # Run for L steps.
    for step in range(L):
      q = q + step_size*p
      if step != L-1:
        p = p - step_size * tf.gradients(energy_fn(q), q)[0]
    proposed_U = energy_fn(q)
    p = p - step_size * tf.gradients(proposed_U, q)[0] / 2.
    p = -p
    proposed_K = tf.reduce_sum(tf.square(p)) / 2.
    
    p = tf.debugging.check_numerics(p, "Nans in p.")
    q = tf.debugging.check_numerics(q, "Nans in q.")
   
    accept = tf.random.uniform([]) < tf.exp(init_U - proposed_U + init_K - proposed_K)
    accept_samples = tf.logical_and(accept, i > burn_in)
    samples = tf.cond(accept_samples, lambda: samples.write(num_accepted, q), lambda: samples)
    accept_samples = tf.squeeze(accept_samples)
    q = tf.cond(accept, lambda: q, lambda: init_q)
    return i+1, num_accepted + tf.to_int32(accept_samples), q, samples
    
  def hmc_predicate(i, num_accepted, unused_q, unused_samples):
    return tf.logical_and(tf.less(i, burn_in + max_steps), 
                          tf.less(num_accepted, num_samples*thinning_steps))
  
  results = tf.while_loop(hmc_predicate,
                          hmc_step,
                          (0, 0, init_X, samples), back_prop=False)
  #[num_samples, data_dim]
  samples = results[-1].stack()
  samples = tf.reshape(samples, [num_samples, thinning_steps,-1])
  samples = samples[:,-1,:] 

  num_steps = results[0]
  num_accepted = results[1]
  accept_ratio = num_accepted / (num_steps - burn_in)
  tf.summary.scalar("acceptance_ratio", accept_ratio)
  tf.summary.scalar("num_hmc_steps", num_steps - burn_in)
  return samples


def langevin(energy_fn, init_X, step_size=1.0, burn_in=100, num_samples=1000, thinning_steps=1, max_steps=None):

  samples = tf.TensorArray(init_X.dtype, 
                           size=num_samples*thinning_steps, 
                           dynamic_size=False, 
                           name='samples_ta')
  init_X = init_X[tf.newaxis,:] 
  X_shape = tf.shape(init_X)
  
  if max_steps == None:
    max_steps = 1000*num_samples*thinning_steps

  def langevin_step(i, num_accepted, X, samples):
    # Compute initial kinetic and potential energies.
    grad_X = tf.gradients(energy_fn(X), X)[0]
    
    new_X = X + step_size*grad_X + tf.math.sqrt(2*step_size)*tf.random.normal(X_shape)
   
    new_X = tf.debugging.check_numerics(new_X, "Nans in X.")
   
    accept = tf.squeeze(i > burn_in)
    samples = tf.cond(accept, lambda: samples.write(num_accepted, new_X), lambda: samples)
    new_X = tf.cond(accept, lambda: new_X, lambda: X)
    return i+1, num_accepted + tf.to_int32(accept), new_X, samples
    
  def langevin_predicate(i, num_accepted, unused_X, unused_samples):
    # We're not done as long as we haven't gone over the max number of steps
    # and we haven't accepted enough samples.
    return tf.logical_and(tf.less(i, burn_in + max_steps), 
                          tf.less(num_accepted, num_samples*thinning_steps))
  
  results = tf.while_loop(langevin_predicate,
                          langevin_step,
                          (0, 0, init_X, samples), back_prop=False)
  #[num_samples, data_dim]
  samples = results[-1].stack()
  samples = tf.reshape(samples, [num_samples, thinning_steps,-1])
  samples = samples[:,-1,:] 

  num_steps = results[0]
  num_accepted = results[1]
  accept_ratio = num_accepted / (num_steps - burn_in)
  #tf.summary.scalar("acceptance_ratio", accept_ratio)
  #tf.summary.scalar("num_langevin_steps", num_steps - burn_in)
  return samples
