# RFHO
Gradient-based hyperparameter optimization package with 
[TensorFlow](https://www.tensorflow.org/)

The package implements the three algorithms presented in the paper
 _Forward and Reverse Gradient-Based Hyperparameter Optimization_ [2017]
 (https://arxiv.org/abs/1703.01785):
- Reverse-HO, generalization of algorithms presented in Domke [2012] and MacLaurin et Al. [2015] (without reversable dynamics and "reversable dtype")
- Forward-HO
- Real-Time Hyperparameter Optimization (RTHO)

## Installation & dependencies

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


#### Quick start 

- [Self contained example](https://github.com/lucfra/RFHO/blob/master/rfho/examples/RFHO%20starting%20example.ipynb) on MNIST (!) with Reverse-HO 
(Forward-HO and RTHO coming very soon..)
- [A module with a more complete set of examples](https://github.com/lucfra/RFHO/blob/master/rfho/examples/all_methods_on_mnist.py) 
showing all algorithms an various models (still on MNIST...)

#### Core idea

The objective is to minimize some validation function _E_ with respect to
 a vector of hyperparameters _lambda_. The validation error depends on the model output and thus
 on the model parameters _w_. 
  _w_ should be a minimizer of the training error and the hyperparameter optimization 
  problem can be natuarlly formulated as a __bilevel optimization__ problem.  
   Since these problems are rather hard to tackle, we  
explicitly take into account the learning dynamics used to obtain the model  
parameters (e.g. you can think about stochastic gradient descent with momentum),
and we formulate
HO as a __constrained optimization__ problem. See the [paper]((https://arxiv.org/abs/1703.01785)) for details.

#### Code structure

- All the hyperparameter optimization algorithms are implemented in the module `hyper_gradients`.
The classes `ReverseHyperGradient` and `ForwardHyperGradient` are responsible 
of the computation of the hyper-gradients while `RealTimeHO` is an helper class
that implements RTHO algorithm (based on `ForwardHyperGradient`).
- The module `optimizers` contains classes that implement 
gradient descent based iterative optimizers. Since 
the HO methods need to access to the optimizer dynamics (which is seen as 
a dynamical system) we haven't been able to employ TensorFlow optimizers. 
At the moment the following optimizers are implemented
    - `GradientDescentOptimizer`
    - `MomentumOptimizer`
    - `AdamOptimizer` (not compatible yet with Forward-HO)
- `models` module contains some helper function to build up models. It also 
contains the core function `vectorize_model` which transform the computational
graph so that all the parameters of the model are conveniently collected into 
a single vector (rank-1 tensor) of the appropriate dimension (see method doc
for further details)
- `utils` method contains some use useful functions. Most notably `cross_entropy_loss`
 re-implements the cross entropy with softmax output. This was necessary since 
 tensorflow function has zero Hessian.
