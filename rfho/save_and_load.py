import matplotlib.pyplot as plt
from IPython.display import IFrame
import IPython
import gzip
import os
import _pickle as pickle
import numpy as np


settings = {
    'NOTEBOOK_TITLE': ''
}


def check_or_create_dir(directory, create=True):
    if create:
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
    if settings['NOTEBOOK_TITLE']:
        directory += '/' + settings['NOTEBOOK_TITLE']
        if create:
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
    return directory


def save_fig(name, default_overwrite=False):
    directory = check_or_create_dir('plots')

    filename = directory + '/%s.pdf' % name
    if not default_overwrite and os.path.isfile(filename):
        IPython.display.display(IFrame(filename, width=800, height=600))
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
    plt.savefig(filename)
    print('file saved')


def save_obj(obj, name, default_overwrite=False):
    directory = check_or_create_dir('obj_data')

    filename = directory + '/%s.pkgz' % name
    if not default_overwrite and os.path.isfile(filename):
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
        print('Overwriting...')
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print('File saved!')


def load_obj(name):
    directory = check_or_create_dir('obj_data', create=False)

    filename = directory + '/%s.pkgz' % name
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def save_adjacency_matrix_for_gephi(matrix, name, class_names=None):
    directory = check_or_create_dir('GePhi_adj_mat')

    filename = directory + '/%s.csv' % name

    m, n = np.shape(matrix)
    assert m == n, '%s should be a square matrix.' % matrix
    if not class_names:
        class_names = [str(k) for k in range(n)]

    left = np.array([class_names]).T
    matrix = np.hstack([left, matrix])
    up = np.vstack([[''], left]).T
    matrix = np.vstack([up, matrix])

    np.savetxt(filename, matrix, delimiter=';', fmt='%s')
