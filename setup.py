from distutils.core import setup

setup(
    name='RFHO',
    version='0.1',
    packages=['rfho', 'rfho.experiments'],
    # TODO have to decide what to do with things in experiments. experiments.common is a useful
    # class but maybe its misplaced.
    url='',
    license='',
    author='Luca Franceschi',
    author_email='igor_mio@hotmail.it',
    description='Gradient based hyperparameter optimization package ', requires=['tensorflow', 'matplotlib']
)
