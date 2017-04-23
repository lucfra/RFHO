# RFHO
Gradient-based hyperparameter optimization package with TensorFlow

The package implements the three algorithms presented in the paper _Forward and Reverse Gradient-Based Hyperparameter Optimization_ [2017] (https://arxiv.org/abs/1703.01785):
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
TensorFlow, thus making them readily applicable to deep learning systems. The package is under development and the code
is not particularly optimized,
please feel free to issues comments, suggestions and feedbacks! You can also email me at luca.franceschi@iit.it .

#### Core idea

The objective is to minimize some validaiton funcition $E$ with respect to
 a vector of hyperparameters $\lambda$. The validation error should be calculated
 at a minimizer 
 Instead of treating hyperparameter optimization as a _bilevel optimization_ problem, we
explicitly take into account the training dynamics, and we formulate
it as a _constrained optimization_ problem.

#### Modules

- All the algorithms are implemented in the module `hyper_gradients`. The classes `ReverseHyperGradient` and `ForwardHyperGradient` initialize
