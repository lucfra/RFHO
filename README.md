# RFHO
Gradient-based hyperparameter optimization package with 
[TensorFlow](https://www.tensorflow.org/)

The package implements the three algorithms presented in the paper
 _Forward and Reverse Gradient-Based Hyperparameter Optimization_ [2017]
 (https://arxiv.org/abs/1703.01785). 
- Reverse-HG, generalization of algorithms presented in Domke [2012] and MacLaurin et Al. [2015] (without reversable dynamics and "reversable dtype")
- Forward-HG
- Real-Time Hyperparameter Optimization (RTHO)

The first two algorithms compute, with different procedures, the gradient
  of a validation error with respect to hyperparameters, while the last, based on Forward-HG, 
  performs "real time" (i.e. at training time) hyperparameter updates.

## Installation & Dependencies

Clone the repository and run setup script.

```
git clone https://github.com/lucfra/RFHO.git
cd rfho
python setup.py install
```

Beside "usual" packages (`numpy`, `pickle`, `gzip`), RFHO depends on `tensorflow`. Some secondary module depends also
on `cvxopt` (projections) and `intervaltree`. The core code works without this packages, so feel free to ignore
 these requirements.

Please note that required packages will not be installed automatically.

## Overview

Aim of this package is to implement and develop gradient-based hyperparameter optimization (HO) techniques in
TensorFlow, thus making them readily applicable to deep learning systems. The package is under
development and at the moment the code
is not particularly optimized;
please feel free to issues comments, suggestions and feedbacks! You can also email me at luca.franceschi@iit.it .


#### Quick Start 

- [Self contained example](https://github.com/lucfra/RFHO/blob/master/rfho/examples/RFHO%20starting%20example.ipynb) on MNIST (!) with `ReverseHG`
- [A module with a more complete set of examples with an `experiment` function
full of parameters](https://github.com/lucfra/RFHO/blob/master/rfho/examples/all_methods_on_mnist.py) 
showing all algorithms an various models (again on MNIST...)

#### Core Steps

- Create a model as you prefer<sup>1</sup> with TensorFlow,
- obtain a vector representation of your model with the function 
`rfho.vectorize_model`,
- define the hyperparameters you wish to optimize<sup>2</sup> as `tensorflow.Variable`,
- define a training and a validation error as scalar `tensorflow.Tensor`,
- create a training dynamics with a subclass of `rfho.Optimizer` (at the moment
gradient descent,
gradient descent with momentum and Adam algorithms are available),
- chose and hyper-gradient computation algorithm among
`rfho.ForwardHG` and `rfho.ReverseHG` (see next section) and 
instantiate `rfho.HyperOptimizer`,
- execute `rfho.HyperOptimizer.run` function inside a `tensorflow.Session`
and optimize both the parameter and 
hyperparameter of your model (learning rate included)!


```python
import rfho as rf
import tensorflow as tf

model = create_model(...)  
w, out = rf.vectorize_model(model.var_list, model.out)

lambda1 = tf.Variable(...)
lambda2 = tf.Variable(...)
training_error = J(w, lambda1, lambda2)
validation_error = f(w)

lr = tf.Variable(...)
training_dynamics = rf.GradientDescentOptimizer.create(w, lr=lambda1, loss=training_error)

hyper_dict = {validation_error: [lambda1, lambda2, lr]}
hyper_opt = rf.HyperOptimizer(training_dynamics, hyper_dict, method=rf.ForwardHG)

hyper_batch_size = 100
with tf.Session().as_default():
    hyper_opt.initialize()  # initializing just once corresponds to RTHO algorithm
    for k in range(...):
        hyper_opt.run(hyper_batch_size, ....)  
```
____
<sup>1</sup> This is gradient-based optimization and for the computation
of the hyper-gradients second order derivatives of the training error show up
(_even tough no Hessian is explicitly computed at any time_);
therefore, all the ops used
in the model should have a second order derivative registered in `tensorflow`.

<sup>2</sup> For the hyper-gradients to make sense, hyperparameters should be 
real-valued. Moreover, while `ReverseHG` should handle generic r-rank tensor 
hyperparameters (_tested on scalars, vectors and matrices_), in `ForwardHG` 
hyperparameters should be scalars or vectors.

#### Which Algorithm Do I Choose?

It's a matter of time versus memory (RAM)!

![alt text](https://github.com/lucfra/RFHO/blob/master/rfho/examples/time_memory.png "mah")


#### The Idea Behind

The objective is to minimize some validation function _E_ with respect to
 a vector of hyperparameters _lambda_. The validation error depends on the model output and thus
 on the model parameters _w_. 
  _w_ should be a minimizer of the training error and the hyperparameter optimization 
  problem can be naturally formulated as a __bilevel optimization__ problem.  
   Since these problems are rather hard to tackle, we  
explicitly take into account the learning dynamics used to obtain the model  
parameters (e.g. you can think about stochastic gradient descent with momentum),
and we formulate
HO as a __constrained optimization__ problem. See the [paper](https://arxiv.org/abs/1703.01785) for details.

#### Code Structure

- All the hyperparameter optimization-related algorithms are implemented in the module `hyper_gradients`.
The classes `ReverseHG` and `ForwardHG` are responsible 
for the computation of the hyper-gradients. `HyperOptimizer` is an interface class
that seamlessly allows for the hyperparameter optimization both in real-time (RTHO algorithm, 
based on `ForwardHG` and _experimental (needs more testing) Truncated-Reverse based on Reverse-HG_) and in "batch"
mode.
- The module `optimizers` contains classes that implement 
gradient descent based iterative optimizers. Since 
the HO methods need to access to the optimizer dynamics (which is seen as 
a dynamical system) we haven't been able to employ TensorFlow optimizers. 
At the moment the following optimizers are implemented
    - `GradientDescentOptimizer`
    - `MomentumOptimizer`
    - `AdamOptimizer` (now compatible with `ForwardHG`)
- `models` module contains some helper function to build up models. It also 
contains the core function `vectorize_model` which transform the computational
graph so that all the parameters of the model are conveniently collected into 
a single vector (rank-1 tensor) of the appropriate dimension (see method doc
for further details)
- `utils` method contains some useful functions. Most notably `cross_entropy_loss`
 re-implements the cross entropy with softmax output. This was necessary since 
`tensorflow.nn.softmax_cross_entropy_with_logits` function has no registered second derivative.

### Citing 

If you use this, please cite the paper.