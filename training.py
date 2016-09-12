import sys
import collections

import tensorflow as tf


ADAM = 'adam'
SGD = 'sgd'
RMSPROP = 'rmsprop'
ADADELTA = 'adadelta'
ADAGRAD = 'adagrad'
MOMENTUM = 'momentum'
NESTEROV = 'nesterov'


class Optimizer(object):
    """Optimizer class to encapsulate (all) optimizers from its creation.
       This is required to enable delayed-build of the optimizer.
    """
    def __init__(self, optimizer_name, initial_lr,
                 step_interval=sys.maxint, rate=1.0, staircase=True):
        """Creates an optimizer with its default hyperparams.
           Note: Momentum-based optimizers (RMSProp, Momentum, Nesterov) should
                 sets its momentum explicitely.
        Parameters
        ----------
        optimizer_name: str
            The optimizer to use. Use keys such as 'tt.training.ADAM' or 'adam'.
        initial_lr: float
            The inital learning rate > 0.
        step_interval: int, optional
            The number of steps when to decay the learning rate.
            Use sys.maxint to use no decay.
        rate: float, optional
            The decay rate.
        staircase: Boolean, optional
            Whether to use staircase decay (default) or not.
        """
        assert initial_lr > 0, "Learning rate must be positive."
        assert step_interval > 0, "Decay step interval must be > 0."
        assert rate > 0 and rate <= 1, "Decay rate must be in range (0, 1]."
        
        self._optimizer_name = optimizer_name.lower()
        self._initial_lr = initial_lr
        
        # set decay
        self._decay = collections.namedtuple("decay", ("step_interval",
                                                       "rate",
                                                       "staircase"))
        self._decay.step_interval = step_interval
        self._decay.rate = rate
        self._decay.staircase = staircase
        
        # set default hyper-params
        self._hyper = collections.namedtuple("hyperparams", ("rho",
                                                             "epsilon",
                                                             "initial_accumulator_value",
                                                             "beta1",
                                                             "beta2",
                                                             "decay",
                                                             "momentum"))
        self.set_hyperparams()
        
    def set_hyperparams(self, epsilon=1e-8, beta1=0.9, beta2=0.999, momentum=0.0,
                        decay=0.9, rho=0.95, initial_accumulator_value=0.1):
        """Sets the hyper parameters. Choose the related ones to your defined
           learning algorithm. All others will be ignored. This function resets
           all not-specified values to its defaults.
        epsilon: float, optional
            The fuzz factor. Value is typically close to 0.
        beta1: float, optional
            The exponential decay rate for the 1st moment estimates.
            Value is typically close to 1.
        beta2: float, optoinal
            The exponential decay rate for the 2nd moment estimates.
            Value is typically close to 1.
        momentum: float, optional
            The momentum to use.
        decay: float, optional
            Discounting factor for the history/coming gradient.
        rho: float, optional
            The rho-decay rate.
        initial_accumulator_value: float, optional
            Starting value for the accumulators, must be positive
        """
        assert epsilon >= 0, "Epsilon must be >= 0 (usually very small)."
        assert beta1 > 0 and beta1 < 1, "Beta1 must be in range (0, 1)."
        assert beta2 > 0 and beta2 < 1, "Beta2 must be in range (0, 1)."
        assert momentum >= 0, "Momentum must be >= 0."
        assert decay >= 0, "Decay must be >= 0."
        assert rho >= 0, "Rho must be >= 0."
        assert initial_accumulator_value >= 0, "Accumulator value must be >= 0."
        
        self._hyper.epsilon = epsilon
        self._hyper.beta1 = beta1
        self._hyper.beta2 = beta2
        self._hyper.momentum = momentum
        self._hyper.decay = decay
        self._hyper.rho = rho
        self._hyper.initial_accumulator_value = initial_accumulator_value
        
    def build(self, global_step):
        """Actually builds the optimizer including the learning rate decay
           if it was configured.
        Parameters
        ----------
        global_step: int or tf.Variable
            The global step counter.
        Returns
        ----------
        Tuple (optimizer, learning_rate) of the created optimizer.
        """
        if self.uses_decay:
            # Decay the learning rate exponentially based on the number of steps
            lr = tf.train.exponential_decay(self.initial_lr,
                                            global_step,
                                            self.decay.step_interval,
                                            self.decay.rate,
                                            staircase=self.decay.staircase)
        else:
            lr = self.initial_lr
        
        if self.name == SGD:
            opt = tf.train.GradientDescentOptimizer(lr)
        elif self.name == ADAM:
            opt = tf.train.AdamOptimizer(lr,
                                         beta1=self._hyper.beta1,
                                         beta2=self._hyper.beta2,
                                         epsilon=self._hyper.epsilon)
        elif self.name == RMSPROP:
            opt = tf.train.RMSPropOptimizer(lr,
                                            decay=self._hyper.decay,
                                            momentum=self._hyper.momentum,
                                            epsilon=self._hyper.epsilon)
        elif self.name == ADADELTA:
            opt = tf.train.AdadeltaOptimizer(lr,
                                             rho=self._hyper.rho,
                                             epsilon=self._hyper.epsilon)
        elif self.name == ADAGRAD:
            opt = tf.train.AdagradOptimizer(lr,
                                            initial_accumulator_value=self._hyper.initial_accumulator_value)
        elif self.name == MOMENTUM:
            opt = tf.train.MomentumOptimizer(lr,
                                             momentum=self._hyper.momentum,
                                             use_nesterov=False)
        elif self.name == NESTEROV:
            opt = tf.train.MomentumOptimizer(lr,
                                             momentum=self._hyper.momentum,
                                             use_nesterov=True)
        else:
            raise ValueError("Unknown optimizer. Contributors welcome...")
        return opt, lr
        
    @property
    def name(self):
        """Gets the optimizers name."""
        return self._optimizer_name
    
    @property
    def initial_lr(self):
        """Gets the initial learning rate."""
        return self._initial_lr
       
    @property
    def decay(self):
        """Gets the (exponential) decay properties."""
        return self._decay
    
    @property
    def uses_decay(self):
        """Indicates whether (exponential) decay is used or not."""
        return True if self.decay.step_interval == sys.maxint or self.decay.rate == 1 else False
    
    @property
    def hyperparams(self):
        """Gets the hyper parameters for the optimizer.
           Only the hyper params relevant for this optimizer are used.
        """
        return self._hyper



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