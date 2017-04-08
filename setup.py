from distutils.core import setup

setup(
    name='RFHO',
    version='0.1',
    packages=['rfho', 'rfho.tests', 'rfho.experiments'],
    url='',
    license='',
    author='lfranceschi',
    author_email='igor_mio@hotmail.it',
    description='Gradient based hyperparameter optimization package ', requires=['tensorflow']
)
