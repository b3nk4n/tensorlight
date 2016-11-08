<div align="center">
  <img src="https://github.com/bsautermeister/tensorlight/blob/master/assets/tensorlight.png" alt="TensorLight"><br>
</div>
-----------------

TensorLight is a high-level framework for [TensorFlow](https://www.tensorflow.org/)-based machine intelligence applications. It reduces boilerplate code and enables advanced features that are not yet provided out-of-the-box.


### Guiding Principles

The TensorLight framework is developed under its four core principles:

- **Simplicity:** Straight-forward to use for anybody who has already worked with TensorFlow. Especially, no further learning is required regarding how to define a model's graph definition.
- **Compactness:** Reduce boilerplate code, while keeping the transparency and flexibility of TensorFlow. 
- **Standardization:** Provide a standard way in respect to the implementation of models and datasets in order to save time. Further, it automates the whole training and validation process, but also provides hooks to maintain customizability.
- **Superiority:** Enable advanced features that are not included in the TensorFlow API, as well as retain its full functionality.


### Key Features

To highlight the advanced features of TensorLight, an incomplete list of some main functionalities is provided that are not shipped with TensorFlow by default, or might even be missing in other high-level APIs. These include:

- Transparent lifecycle management of the session and graph definition.
- Abstraction of models and datasets to provide a reusable plug-and-play support.
- Effortless support to train a model symmetrically on multiple GPUs, as well as prevent TensorFlow to allocate memory on other GPU devices of the cluster.
- Train or evaluate a model with a single line of code.
- Abstracted, runtime-exchangeable input pipelines which either use the simple feeding mechanism with NumPy arrays, or even multi-threaded input queues.
- Automatic saving and loading of hyperparameters as JSON to simplify the evaluation management of numerous trainings.
- Ready-to-use loss functions and metrics, even with latest advances for perceptual motivated image similarity assessment.
- Extended recurrent functions to enable scheduled sampling, as well as an implementation of a ConvLSTM cell.
- Automatic creation of periodic checkpoints and TensorBoard summaries.
- Ability to work with other higher-level libraries hand in hand, such as *tf.contrib* or *TF-slim*.


### Architecture

From an architectural perspective, the framework can be split into three main components. First, a collection of *utility function* that are unrelated to machine learning. Examples are functions to download and extract datasets, to process images and videos, or to generate animated GIFs and videos from a data array, to name just a few. Second, the *high-level library* which builds on top of TensorFlow. It includes several modules that either provide a simple access to functionally that it repeatedly required when developing deep learning applications, or features that are not included in TensorFlow yet. For instance, it handles the creation of weight and bias variables internally, offers a bunch of ready-to-use loss and initialization functions, or comes with some advanced visualization features to display feature maps or output images directly in an IPython Notebook. Third, an *abstraction layer* to simplify the overall lifecycle, to generalize the definition of a model graphs, as well as to enable a reusable and consistent access to datasets.

<div align="center">
  <img src="https://github.com/bsautermeister/tensorlight/blob/master/assets/tensorlight_arch.png" alt="TensorLight Architecture" width="600"><br>
</div>

The user program can either exploit the high-level library and the provided utility functions for his existing projects, or take advantage from TensorLight's abstraction layes while creating new deep learning applications. The latter enables to radically reduce the amount of code that has to be written for training or evaluating the model. This is realized by encapsulating the lifecycle of TensorFlow's session, graph, summary-writer or checkpoint-saver, as well as the entire training or evaluation loop within a *runtime module*.


### Examples

You want to learn more? Check out the [tutorial and code examples](examples/README.md).
