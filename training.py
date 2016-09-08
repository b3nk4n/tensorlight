import tensorflow as tf

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers
    in a mulit-GPU environment.
    Note that this function provides a synchronization point across all towers.
    Parameters
    ----------
    tower_grads: List of lists of (gradient, variable) tuples
        The tower gradients. The outer list is over individual gradients.
        The inner list is over the gradient calculation for each tower.
    Returns
    ----------
    average_grads: List of pairs of (gradient, variable)
        The gradients where the gradient has been averaged
        across all towers.
    """
    with tf.name_scope("avg_grads"):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    

def inverse_sigmoid_decay(decay_variable, global_step, decay_rate=1000.0,
                          name=None):
    """Applies inverse sigmoid decay to the decay variable (learning rate).
    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies an inverse sigmoid decay 
    function to a provided initial variable value.  It requires a `global_step`
    value to compute the decayed variable value. You can just pass a TensorFlow
    variable that you increment at each training step.
    The function returns the decayed variable value.  It is computed as:
    
    With decay-var = 1.0, gstep = x, decay_rate = 10000.0
    1.0*(10000.0/(10000.0+exp(x/(10000.0))))
    
    ```python
    decayed_var = decay_variable *
                  decay_rate / (decay_rate + exp(global_step / decay_rate))                         
    ```
    
    Rough Infos           | Value @ t=0 | (Real) decay start | Reaches Zero   
    -------------------------------------------------------------------------
    decay_rate:    10.0   | 0.9         |          -40       |         100
    decay_rate:   100.0   | 0.985       |          -20       |       1,100
    decay_rate:  1000.0   | 1.0         |        2,000       |      12,000
    decay_rate: 10000.0   | 1.0         |       50,000       |     110,000
    
    Parameters
    ----------
    decay_variable: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial variable value to decay.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate >> 1.
    name: String.  Optional name of the operation.  Defaults to 
      'InvSigmoidDecay'
    Returns
    ----------
    A scalar `Tensor` of the same type as `decay_variable`.  The decayed
    variable value (such as learning rate).
    """
    assert decay_rate > 1, "The decay_rate has to be >> 1."
    
    with ops.name_scope(name, "InvSigmoidDecay",
                        [decay_variable, global_step,
                         decay_steps, decay_rate]) as name:
        decay_variable = ops.convert_to_tensor(decay_variable, name="decay_variable")
        dtype = decay_variable.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        decay_rate = math_ops.cast(decay_rate, dtype)
        
        denom = decay_rate + tf.exp(global_step / decay_rate)
        return math_ops.mul(decay_variable, decay_rate / denom, name=name)