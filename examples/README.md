## Tutorial and Code Examples

Several code examples are provided above. The following example explains the training and evaluation of a convolutional autoencoder, which reconstructs the entire input sequence of Moving MNIST. It emphasizes the simplicity of deep learning applications implemented with TensorLight.

### Model

At first, the model has to be defined. This is realized by implementing a class that derives from `light.model.AbstractModel`. This class provides a standard interface to define the graph structure, the loss layer, as well as how this model has to be evaluated by the framework.

```python
import tensorflow as tf
import tensorlight as light
from tf.contrib.layers import *

class ConvAutoencoderModel(light.model.AbstractModel):    
  def __init__(self, weight_decay=0.0):
    super(ConvAutoencoderModel, self).__init__(weight_decay)
        
  @light.utils.attr.override
  def inference(self, inputs, targets, feeds, is_train, device_scope, memory_device):
    with tf.variable_scope("Encoder"):
      conv1 = light.network.conv2d("Conv1", inputs, 4, (5, 5), (2, 2),
                                   weight_init=xavier_initializer_conv2d(),
                                   regularizer=l2_regularizer(self.weight_decay),
                                   activation=tf.nn.relu)
      conv1_bn = batch_norm(conv1, is_training=is_train, scope="conv1_bn")
      conv2 = light.network.conv2d("Conv2", conv1_bn, 8, (3, 3), (2, 2),
                                   weight_init=xavier_initializer_conv2d(),
                                   regularizer=l2_regularizer(self.weight_decay),
                                   activation=tf.nn.relu)
      conv2_bn = batch_norm(conv2, is_training=is_train, scope="conv2_bn")
      learned_rep = conv2_bn
      
    with tf.variable_scope("Decoder"):
      convt = light.network.conv2d_transpose("Convt1", learned_rep, 4, (3, 3), (2, 2),
                                             weight_init=light.init.bilinear_initializer(),
                                             regularizer=l2_regularizer(self.weight_decay),
                                             activation=tf.nn.relu)
      convt_bn = batch_norm(convt, is_training=is_train, scope="convt_bn")
      return light.network.conv2d_transpose("Convt2", convt_bn, 1, (5, 5), (2, 2),
                                            weight_init=light.init.bilinear_initializer(), 
                                            regularizer=l2_regularizer(self.weight_decay),
                                            activation=tf.nn.sigmoid)
                                              
  @light.utils.attr.override
  def loss(self, predictions, targets, device_scope):
    return light.loss.bce(predictions, targets)
    
  @light.utils.attr.override
  def evaluation(self, predictions, targets, device_scope):
    psnr = light.image.psnr(predictions, targets)
    sharpdiff = light.image.sharp_diff(predictions, targets)
    ssim = light.image.ssim(predictions, targets)
    return {"psnr": psnr, "sharpdiff": sharpdiff, "ssim": ssim}
```

In this case, the model is trained using binary cross-entropy. Moreover, image similarity metrics like PSNR, sharpness difference and SSIM are going be calculated and visualized in TensorBoard for every validation iteration in addition to the error value of the objective function.


### Training

The program code of an entire training process is explained next. Within the context of a runtime, the model, optimizer and datasets are registered with its parameters. Each runtime and dataset instance has a path-parameter in order to specify the root directory of the training outputs and meta files, as well as to reuse downloaded and preprocessed files across multiple processes. Afterwards, the computation graph is built and the training loop is started for 100 epochs. Since the example tries to reconstruct the inputs, one can set `autoencoder=True` in order to redirect the target pipeline of the dataset to the input data. Additionally, each batch of size 256 is distributed across two GPUs to speed up training.

```python
import tensorlight as light
from tensorlight.datasets.moving_mnist import *

with light.core.MultiGpuRuntime("/tmp/train", gpu_devices=[0, 1]) as rt:
  rt.register_model(ConvAutoencoderModel(5e-4))
  rt.register_optimizer(light.training.Optimizer("adam", initial_lr=0.001,
                                                 step_interval=1000, rate=0.95))	
  rt.register_datasets(MovingMNISTTrainDataset("/tmp/data", as_binary=True,
                                               input_shape=[10, 64, 64, 1]),
                       MovingMNISTValidDataset("/tmp/data/", as_binary=True,
                                               input_shape=[10, 64, 64, 1]))
  rt.build(is_autoencoder=True)
  rt.train(batch_size=256, epochs=100,
           valid_batch_size=200, validation_steps=1000)
```

When the training is complete, the runtime is ready to do inference or testing on the model. In case the model's performance is still unsatisfactory, it is possible to continue the previous training for a few steps more. In addition, the framework allows to register a validation callback in order to perform specific functionality after every validation process. In this case, a GIF animation is generated that compares the ground truth with its reconstruction of a randomly generated sequence from the validation set.

```python
  # continue training
  def on_valid(rt, global_step):
    inputs, _ = rt.datasets.valid.get_batch(1)
    pred = rt.predict(inputs)
    light.utils.video.write_multi_gif(os.path.join(rt.train_dir, "{}.gif".format(global_step)),
                                      [inputs[0], pred[0]], fps=5)
  rt.train(batch_size=256, steps=10000, valid_batch_size=200,
           validation_steps=1000, on_validate=on_valid)
```

### Evaluation

After the whole training process is complete, the model is ready to get evaluated on the test set. Therefore, model and test dataset instances are created and registered to a new runtime object. When building the model, the last checkpoint file and the model's hyperparameter-meta file can be loaded in order to rebuild the graph with the identical weights and model configuration. Last but not least, the testing process can be started with a single line of code.

```python
with light.core.DefaultRuntime("/tmp/train") as rt:
  rt.register_model(ConvAutoencoderModel())
  rt.register_datasets(test_ds=MovingMNISTTestDataset("/tmp/data", as_binary=True,
                                                      input_shape=[10, 64, 64, 1]))
  rt.build(is_autoencoder=True, restore_model_params=True
           restore_checkpoint=light.core.LATEST_CHECKPOINT)
  rt.test(batch_size=100, epochs=100)
```
