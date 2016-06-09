# TensorTools

TensorTools is a programming library for [TensorFlow](https://www.tensorflow.org/) based projects to reduce poilerplate code.

## Modules

- Network: *Provides convenient functions to simplify existing TensorFlow functions*
- Visualization: *Enables to include TensorBoard graphics inside an iPython script*


## Convetions

This section covers some coding conventions that are used within this framework.

#### Scoping and variable naming

Each function that is creating variables using [tf.get_variable()](https://www.tensorflow.org/api_docs/python/state_ops.html#get_variable) surrounds these calls with an own variable scope using [tf.variable_scope](https://www.tensorflow.org/api_docs/python/state_ops.html#variable_scope). These functions require an obligatory *name_or_scope* parameter.

All other functions to not create a *variable scope*, but can create a *name scope* using [tf.name_scope](https://www.tensorflow.org/api_docs/python/framework.html#name_scope) for improved visualization in TensorBoard.

