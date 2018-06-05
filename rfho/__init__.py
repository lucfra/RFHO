# TODO (luca) - is it a good idea to import with *??

import sys

print('-'*60 + '\n'
      'WARNING: This package is no longer supported.',
      'If you want to reproduce the experiment in the ICML 2017 paper please consider ',
      'installing the package from ICML17 branch',
      'Otherwise please take a look at the newer and (hopefully) easier to use package Far-HO available at:',
      'https://github.com/lucfra/FAR-HO\n', file=sys.stderr, sep='\n', end='-'*60 + '\n')

from rfho.models import *
from rfho.hyper_gradients import *
from rfho.optimizers import *
from rfho.save_and_load import *
from rfho.utils import *
from rfho.datasets import Dataset, ExampleVisiting, Datasets

