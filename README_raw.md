This repository introduces the fundamental concepts of
[PyTorch](https://github.com/pytorch/pytorch)
through self-contained examples.

At its core, PyTorch provides two main features:
- An n-dimensional Tensor, similar to numpy but can run on GPUs
- Automatic differentiation for building and training neural networks

We will use a fully-connected ReLU network as our running example. The network
will have a single hidden layer, and will be trained with gradient descent to
fit random data by minimizing the Euclidean distance between the network output
and the true output.

### Table of Contents
- <a href='#warm-up-numpy'>Warm-up: numpy</a>
- <a href='#pytorch-tensors'>PyTorch: Tensors</a>
- <a href='#pytorch-variables-and-autograd'>PyTorch: Variables and autograd</a>
- <a href='#pytorch-defining-new-autograd-functions'>PyTorch: Defining new autograd functions</a>
- <a href='#tensorflow-static-graphs'>TensorFlow: Static Graphs</a>
- <a href='#pytorch-nn'>PyTorch: nn</a>
- <a href='#pytorch-optim'>PyTorch: optim</a>
- <a href='#pytorch-custom-nn-modules'>PyTorch: Custom nn Modules</a>
- <a href='#pytorch-control-flow--weight-sharing'>PyTorch: Control Flow and Weight Sharing</a>

## Warm-up: numpy

Before introducing PyTorch, we will first implement the network using numpy.

Numpy provides an n-dimensional array object, and many functions for manipulating
these arrays. Numpy is a generic framework for scientific computing; it does not
know anything about computation graphs, or deep learning, or gradients. However
we can easily use numpy to fit a two-layer network to random data by manually
implementing the forward and backward passes through the network using numpy
operations:

```python
:INCLUDE tensor/two_layer_net_numpy.py
```

## PyTorch: Tensors

Numpy is a great framework, but it cannot utilize GPUs to accelerate its
numerical computations. For modern deep neural networks, GPUs often provide
speedups of [50x or greater](https://github.com/jcjohnson/cnn-benchmarks), so
unfortunately numpy won't be enough for modern deep learning.

Here we introduce the most fundamental PyTorch concept: the **Tensor**. A PyTorch
Tensor is conceptually identical to a numpy array: a Tensor is an n-dimensional
array, and PyTorch provides many functions for operating on these Tensors. Like
numpy arrays, PyTorch Tensors do not know anything about deep learning or
computational graphs or gradients; they are a generic tool for scientific
computing.

However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their
numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it
to a new datatype.

Here we use PyTorch Tensors to fit a two-layer network to random data. Like the
numpy example above we need to manually implement the forward and backward
passes through the network:

```python
:INCLUDE tensor/two_layer_net_tensor.py
```

## PyTorch: Variables and autograd

In the above examples, we had to manually implement both the forward and
backward passes of our neural network. Manually implementing the backward pass
is not a big deal for a small two-layer network, but can quickly get very hairy
for large complex networks.

Thankfully, we can use
[automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
to automate the computation of backward passes in neural networks. 
The **autograd** package in PyTorch provides exactly this functionality.
When using autograd, the forward pass of your network will define a
**computational graph**; nodes in the graph will be Tensors, and edges will be
functions that produce output Tensors from input Tensors. Backpropagating through
this graph then allows you to easily compute gradients.

This sounds complicated, it's pretty simple to use in practice. We wrap our
PyTorch Tensors in **Variable** objects; a Variable represents a node in a
computational graph. If `x` is a Variable then `x.data` is a Tensor, and
`x.grad` is another Variable holding the gradient of some scalar value with
respect to `x`.

PyTorch Variables have the same API as PyTorch Tensors: (almost) any operation
that you can perform on a Tensor also works on Variables; the difference is that
using Variables defines a computational graph, allowing you to automatically
compute gradients.

Here we use PyTorch Variables and autograd to implement our two-layer network;
now we no longer need to manually implement the backward pass through the
network:

```python
:INCLUDE autograd/two_layer_net_autograd.py
```

## PyTorch: Defining new autograd functions
Under the hood, each primitive autograd operator is really two functions that
operate on Tensors. The **forward** function computes output Tensors from input
Tensors. The **backward** function receives the gradient of some scalar value
with respect to the output Tensors, and computes the gradient that same scalar
value with respect to the input Tensors.

In PyTorch we can easily define our own autograd operator by defining a subclass
of `torch.autograd.Function` and implementing the `forward` and `backward` functions.
We can then use our new autograd operator by constructing an instance and calling it
like a function, passing Variables containing input data.

In this example we define our own custom autograd function for performing the ReLU
nonlinearity, and use it to implement our two-layer network:

```python
:INCLUDE autograd/two_layer_net_custom_function.py
```

## TensorFlow: Static Graphs
PyTorch autograd looks a lot like TensorFlow: in both frameworks we define
a computational graph, and use automatic differentiation to compute gradients.
The biggest difference between the two is that TensorFlow's computational graphs
are **static** and PyTorch uses **dynamic** computational graphs.

In TensorFlow, we define the computational graph once and then execute the same
graph over and over again, possibly feeding different input data to the graph.
In PyTorch, each forward pass defines a new computational graph.

Static graphs are nice because you can optimize the graph up front; for example
a framework might decide to fuse some graph operations for efficiency, or to
come up with a strategy for distributing the graph across many GPUs or many
machines. If you are reusing the same graph over and over, then this potentially
costly up-front optimization can be amortized as the same graph is rerun over
and over.

One aspect where static and dynamic graphs differ is control flow. For some models
we may wish to perform different computation for each data point; for example a
recurrent network might be unrolled for different numbers of time steps for each
data point; this unrolling can be implemented as a loop. With a static graph the
loop construct needs to be a part of the graph; for this reason TensorFlow
provides operators such as `tf.scan` for embedding loops into the graph. With
dynamic graphs the situation is simpler: since we build graphs on-the-fly for
each example, we can use normal imperative flow control to perform computation
that differs for each input.

To contrast with the PyTorch autograd example above, here we use TensorFlow to
fit a simple two-layer net:

```python
:INCLUDE autograd/tf_two_layer_net.py
```


## PyTorch: nn
Computational graphs and autograd are a very powerful paradigm for defining
complex operators and automatically taking derivatives; however for large
neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the computation
into **layers**, some of which have **learnable parameters** which will be
optimized during learning.

In TensorFlow, packages like [Keras](https://github.com/fchollet/keras),
[TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim),
and [TFLearn](http://tflearn.org/) provide higher-level abstractions over
raw computational graphs that are useful for building neural networks.

In PyTorch, the `nn` package serves this same purpose. The `nn` package defines a set of
**Modules**, which are roughly equivalent to neural network layers. A Module receives
input Variables and computes output Variables, but may also hold internal state such as
Variables containing learnable parameters. The `nn` package also defines a set of useful
loss functions that are commonly used when training neural networks.

In this example we use the `nn` package to implement our two-layer network:

```python
:INCLUDE nn/two_layer_net_nn.py
```


## PyTorch: optim
Up to this point we have updated the weights of our models by manually mutating the
`.data` member for Variables holding learnable parameters. This is not a huge burden
for simple optimization algorithms like stochastic gradient descent, but in practice
we often train neural networks using more sophisiticated optimizers like AdaGrad,
RMSProp, Adam, etc.

The `optim` package in PyTorch abstracts the idea of an optimization algorithm and
provides implementations of commonly used optimization algorithms.

In this example we will use the `nn` package to define our model as before, but we
will optimize the model using the Adam algorithm provided by the `optim` package:

```python
:INCLUDE nn/two_layer_net_optim.py
```


## PyTorch: Custom nn Modules
Sometimes you will want to specify models that are more complex than a sequence of
existing Modules; for these cases you can define your own Modules by subclassing
`nn.Module` and defining a `forward` which receives input Variables and produces
output Variables using other modules or other autograd operations on Variables.

In this example we implement our two-layer network as a custom Module subclass:

```python
:INCLUDE nn/two_layer_net_module.py
```


## PyTorch: Control Flow + Weight Sharing
As an example of dynamic graphs and weight sharing, we implement a very strange
model: a fully-connected ReLU network that on each forward pass chooses a random
number between 1 and 4 and uses that many hidden layers, reusing the same weights
multiple times to compute the innermost hidden layers.

For this model can use normal Python flow control to implement the loop, and we
can implement weight sharing among the innermost layers by simply reusing the
same Module multiple times when defining the forward pass.

We can easily implement this model as a Module subclass:

```python
:INCLUDE nn/dynamic_net.py
```
