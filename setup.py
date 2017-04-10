from distutils.core import setup

setup(
    name='RFHO',
    version='0.1',
    packages=['rfho'],  # TODO I (Luca) would like to include also experiment package (at least experiments.common)
    url='',
    license='',
    author='Luca Franceschi',
    author_email='igor_mio@hotmail.it',
    description='Gradient based hyperparameter optimization package ', requires=['tensorflow', 'matplotlib']
)
