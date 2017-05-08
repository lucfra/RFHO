import time
from collections import OrderedDict
from functools import reduce

import matplotlib.pyplot as plt

from rfho import as_list

try:
    from IPython.display import IFrame
    import IPython
except ImportError:
    print('Looks like IPython is not installed...')
    IFrame, IPython = None, None
import gzip
import os
import _pickle as pickle
import numpy as np


def join_paths(*paths):
    return reduce(lambda acc, new_path: os.path.join(acc, new_path), paths)

settings = {
    'NOTEBOOK_TITLE': ''
}


FOLDER_NAMINGS = {
    'EXP_ROOT': os.path.expanduser(join_paths('~user', 'Experiments')),
    'OBJ_DIR': 'Obj_data',
    'PLOTS_DIR': 'Plots',
    'MODELS_DIR': 'Models',
    'GEPHI_DIR': 'GePhi'
}


def check_or_create_dir(directory, notebook_mode=True, create=True):
    if create:
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
    if notebook_mode and settings['NOTEBOOK_TITLE']:
        directory = join_paths(directory, settings['NOTEBOOK_TITLE'])  # += '/' + settings['NOTEBOOK_TITLE']
        if create:
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
    return directory


def save_fig(name, notebook_mode=True, default_overwrite=False):
    directory = check_or_create_dir(FOLDER_NAMINGS['PLOTS_DIR'],
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory, '%s.pdf' % name)  # directory + '/%s.pdf' % name
    if not default_overwrite and os.path.isfile(filename):
        if IPython is not None:
            IPython.display.display(IFrame(filename, width=800, height=600))
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
    plt.savefig(filename)
    print('file saved')


def save_obj(obj, name, notebook_mode=True, default_overwrite=False):
    directory = check_or_create_dir(FOLDER_NAMINGS['OBJ_DIR'], notebook_mode=notebook_mode)

    filename = join_paths(directory, '%s.pkgz' % name)  # directory + '/%s.pkgz' % name
    if not default_overwrite and os.path.isfile(filename):
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
        print('Overwriting...')
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)
        print('File saved!')


def load_obj(name, notebook_mode=False):
    directory = check_or_create_dir(FOLDER_NAMINGS['OBJ_DIR'], notebook_mode=notebook_mode, create=False)

    filename = directory + '/%s.pkgz' % name
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def save_adjacency_matrix_for_gephi(matrix, name, notebook_mode=True, class_names=None):
    directory = check_or_create_dir(FOLDER_NAMINGS['GEPHI_DIR'], notebook_mode=notebook_mode)

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


def save_setting(local_variables, excluded=None, default_overwrite=False, collect_data=True,
                 notebook_mode=True, do_print=True, append_string=''):
    dictionary = generate_setting_dict(local_variables, excluded=excluded)
    if do_print:
        print('SETTING:')
        for k, v in dictionary.items():
            print(k, v, sep=': ')
        print()
    if collect_data: save_obj(dictionary, 'setting' + append_string, default_overwrite=default_overwrite,
                              notebook_mode=notebook_mode)


def generate_setting_dict(local_variables, excluded=None):
    """
    Generates a dictionary of (name, values) of local variables (typically obtained by vars()) that
    can be saved at the beginning of the experiment. Furthermore, if an object obj in local_variables implements the
    function setting(), it saves the result of obj.setting() as value in the dictionary.

    :param local_variables:
    :param excluded: (optional, default []) variable or list of variables to be excluded.
    :return: A dictionary
    """
    excluded = as_list(excluded) or []
    setting_dict = {k: v.setting() if hasattr(v, 'setting') else v
                    for k, v in local_variables.items() if v not in excluded}
    import datetime
    setting_dict['datetime'] = str(datetime.datetime.now())
    return setting_dict


class Timer:

    _div_unit = {'ms': 1. / 1000,
                 'sec': 1.,
                 'min': 60.,
                 'hr': 3600.}

    def __init__(self, unit='sec', round_off=True):
        self._starting_times = []
        self._stopping_times = []
        self._running = False
        self.round_off = round_off
        assert unit in Timer._div_unit
        self.unit = unit

    def start(self):
        if not self._running:
            self._starting_times.append(time.time())
            self._running = True
        return self

    def stop(self):
        if self._running:
            self._stopping_times.append(time.time())
            self._running = False
        return self

    def raw_elapsed_time_list(self):
        def _maybe_add_last():
            t2 = self._stopping_times if len(self._starting_times) == len(self._stopping_times) else \
                self._stopping_times + [time.time()]
            return zip(self._starting_times, t2)
        return [t2 - t1 for t1, t2 in _maybe_add_last()]

    def elapsed_time(self):
        res = sum(self.raw_elapsed_time_list())/Timer._div_unit[self.unit]
        return res if not self.round_off else int(res)


class Saver:

    def __init__(self, experiment_name, *args, timer=None, root_directory=FOLDER_NAMINGS['EXP_ROOT'],
                 do_print=True, collect_data=True, default_overwrite=False):
        """
        Initialize a saver to collect data. (Intended to be used together with OnlinePlotStream.)

        :param experiment_name: string, name of experiment
        :param args: a list of (from pairs to at most) 5-tuples that represent the things you want to save.
                      The first arg of each tuple should be a string that will be the key of the save_dict.
                      Then there can be either a callable with signature (step) -> None
                      Should pass the various args in ths order:
                          fetches: tensor or list of tensors to compute;
                          feeds (optional): to be passed to tf.Session.run. Can be a
                          callable with signature (step) -> feed_dict
                          options (optional): to be passed to tf.Session.run
                          run_metadata (optional): to be passed to tf.Session.run
        :param timer: optional timer object. If None creates a new one. If false does not register time.
                        If None or Timer it adds to the save_dict an entry time that record elapsed_time.
                        The time required to perform data collection and saving are not counted, since typically
                        the aim is to record the true algorithm execution time!
        :param root_directory: string, name of root directory (default ~HOME/Experiments)
        :param do_print: (optional, default True) will print by default `save_dict` each time method `save` is executed
        :param collect_data: (optional, default True) will save by default `save_dict` each time
                            method `save` is executed
        """
        self.directory = join_paths(root_directory, experiment_name)
        if collect_data: check_or_create_dir(self.directory)

        self.do_print = do_print
        self.collect_data = collect_data
        self.default_overwrite = default_overwrite

        assert isinstance(args[0], str), 'Check args! first arg: %s. Should be a string. All args: %s' % (args[0], args)
        assert isinstance(timer, Timer) or timer is None or timer is False, 'timer param not good...'

        processed_args = []
        k = 0
        while k < len(args):
            part = [args[k]]
            k += 1
            while k < len(args) and not isinstance(args[k], str):
                part.append(args[k])
                k += 1
            assert len(part) >= 2, 'Check args! Last part %s' % part
            part += [None] * (5 - len(part))  # representing name, fetches, feeds, options, metadata
            processed_args.append(part)

        if timer is None:
            timer = Timer()

        self.timer = timer
        self.processed_args = processed_args

    def save(self, step, append_string="", do_print=None, collect_data=None):
        from tensorflow import get_default_session

        if do_print is None: do_print = self.do_print
        if collect_data is None: collect_data = self.collect_data

        ss = get_default_session()
        if ss is None and do_print: print('WARNING, No default session')

        if self.timer: self.timer.stop()
        save_dict = OrderedDict([(pt[0], pt[1](step) if callable(pt[1])
                                 else ss.run(pt[1], feed_dict=pt[2](step) if callable(pt[2]) else pt[2],
                                             options=pt[3], run_metadata=pt[4]))
                                 for pt in self.processed_args])

        if self.timer: save_dict['Elapsed time (%s)' % self.timer.unit] = self.timer.elapsed_time()

        if do_print:
            print('SAVE DICT:')
            for key, v in save_dict.items():
                print(key, v, sep=': ')
            print()
        if collect_data:
            self.save_obj(save_dict, str(step) + append_string)

        if self.timer: self.timer.start()

        return save_dict

    def save_fig(self, name):
        return save_fig(name, default_overwrite=self.default_overwrite, notebook_mode=False)

    def save_obj(self, obj, name):
        return save_obj(obj, name, default_overwrite=self.default_overwrite, notebook_mode=False)

    # def work in progress