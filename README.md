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

- [Self contained example]() on MNIST (!) with Reverse-HO 
(Forward-HO and RTHO coming very soon..)
- [A module with a more complete set of examples]() (still on MNIST...)

#### Core idea

The objective is to minimize some validation function _E_ with respect to
 a vector of hyperparameters _lambda_. The validation error depends on the model output and thus
 on the model parameters _w_. 
  _w_ should be a minimizer of the training error and the hyperparameter optimization 
  problem should thus be formulated as a __bilevel optimization__ problem.
   Instead we 
explicitly take into account the learning dynamics used to obtain the model  
parameters (e.g. you can think about stochastic gradient descent with momentum),
and we formulate
HO as a __constrained optimization__ problem. See the [paper]((https://arxiv.org/abs/1703.01785)) for details.

#### Modules

- All the algorithms are implemented in the module `hyper_gradients`.
The classes `ReverseHyperGradient` and `ForwardHyperGradient` 
- ...
