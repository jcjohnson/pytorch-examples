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
`x.grad` is another Variable holding the gradient of `x` with respect to some
scalar value.

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

# TODO: Describe static vs dynamic

To contrast with the PyTorch autograd example above, here we use TensorFlow to
fit a simple two-layer net:

```python
:INCLUDE autograd/tf_two_layer_net.py
```


## PyTorch: nn

```python
:INCLUDE nn/two_layer_net_nn.py
```


## PyTorch: optim

```python
:INCLUDE nn/two_layer_net_optim.py
```


## PyTorch: Custom nn Modules

```python
:INCLUDE nn/two_layer_net_module.py
```


## PyTorch: Control Flow + Weight Sharing

```python
:INCLUDE nn/dynamic_net.py
```
