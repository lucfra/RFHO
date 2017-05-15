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

_EXP_ROOT_FOLDER = os.getenv('RFHO_EXP_FOLDER')
if _EXP_ROOT_FOLDER is None:
    print('Environment variable RFHO_DATA_FOLDER not found. Current directory will be used')
    _EXP_ROOT_FOLDER = join_paths(os.getcwd(), 'Experiments')
print('Experiment save directory is ', _EXP_ROOT_FOLDER)

FOLDER_NAMINGS = {  # TODO should go into a settings file?
    'EXP_ROOT': _EXP_ROOT_FOLDER,
    'OBJ_DIR': 'Obj_data',
    'PLOTS_DIR': 'Plots',
    'MODELS_DIR': 'Models',
    'GEPHI_DIR': 'GePhi'
}


def check_or_create_dir(directory, notebook_mode=True, create=True):
    if not os.path.exists(directory) and create:
        os.mkdir(directory)
        print('folder ', directory, 'has been created')

    if notebook_mode and settings['NOTEBOOK_TITLE']:
        directory = join_paths(directory, settings['NOTEBOOK_TITLE'])  # += '/' + settings['NOTEBOOK_TITLE']
        if not os.path.exists(directory) and create:
            os.mkdir(directory)
            print('folder ', directory, 'has been created')
    return directory


def save_fig(name, root_dir=None, notebook_mode=True, default_overwrite=False):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['PLOTS_DIR']),
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


def save_obj(obj, name, root_dir=None, notebook_mode=True, default_overwrite=False):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['OBJ_DIR']),
                                    notebook_mode=notebook_mode)

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


def load_obj(name, root_dir=None, notebook_mode=True):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['OBJ_DIR']),
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory,  name if name.endswith('.pkgz') else name+'.pkgz')
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


def save_adjacency_matrix_for_gephi(matrix, name, root_dir=None, notebook_mode=True, class_names=None):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['GEPHI_DIR']),
                                    notebook_mode=notebook_mode)
    filename = join_paths(directory, '%s.csv' % name)

    m, n = np.shape(matrix)
    assert m == n, '%s should be a square matrix.' % matrix
    if not class_names:
        class_names = [str(k) for k in range(n)]

    left = np.array([class_names]).T
    matrix = np.hstack([left, matrix])
    up = np.vstack([[''], left]).T
    matrix = np.vstack([up, matrix])

    np.savetxt(filename, matrix, delimiter=';', fmt='%s')


def save_setting(local_variables, root_dir=None, excluded=None, default_overwrite=False, collect_data=True,
                 notebook_mode=True, do_print=True, append_string=''):
    dictionary = generate_setting_dict(local_variables, excluded=excluded)
    if do_print:
        print('SETTING:')
        for k, v in dictionary.items():
            print(k, v, sep=': ')
        print()
    if collect_data: save_obj(dictionary, 'setting' + append_string,
                              root_dir=root_dir,
                              default_overwrite=default_overwrite,
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

    def __init__(self, experiment_name, *items, timer=None, root_directory=FOLDER_NAMINGS['EXP_ROOT'],
                 do_print=True, collect_data=True, default_overwrite=False):
        """
        Initialize a saver to collect data. (Intended to be used together with OnlinePlotStream.)

        :param experiment_name: string, name of experiment
        :param items: a list of (from pairs to at most) 5-tuples that represent the things you want to save.
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
        if collect_data:
            check_or_create_dir(root_directory, notebook_mode=False)
            check_or_create_dir(self.directory, notebook_mode=False)

        self.do_print = do_print
        self.collect_data = collect_data
        self.default_overwrite = default_overwrite

        assert isinstance(timer, Timer) or timer is None or timer is False, 'timer param not good...'

        self.processed_items = []
        self.add_items(*items)

        if timer is None:
            timer = Timer()

        self.timer = timer

    def add_items(self, *items):
        """
        Add items to the save dictionary

        :param items: a list of (from pairs to at most) 5-tuples that represent the things you want to save.
                      The first arg of each tuple should be a string that will be the key of the save_dict.
                      Then there can be either a callable with signature (step) -> None
                      Should pass the various args in ths order:
                          fetches: tensor or list of tensors to compute;
                          feeds (optional): to be passed to tf.Session.run. Can be a
                          callable with signature (step) -> feed_dict
                          options (optional): to be passed to tf.Session.run
                          run_metadata (optional): to be passed to tf.Session.run
        :return: None
        """
        assert len(items) == 0 or isinstance(items[0], str), 'Check items! first arg %s. Should be a string.' \
                                                             'All args: %s' % (items[0], items)

        processed_args = []
        k = 0
        while k < len(items):
            part = [items[k]]
            k += 1
            while k < len(items) and not isinstance(items[k], str):
                part.append(items[k])
                k += 1
            assert len(part) >= 2, 'Check args! Last part %s' % part
            part += [None] * (5 - len(part))  # representing name, fetches, feeds, options, metadata
            processed_args.append(part)
        self.processed_items += processed_args

    def save(self, step, append_string="", do_print=None, collect_data=None):
        """
        Builds and save a dictionary with the keys and values specified at construction time or by method
        `add_items`

        :param step: (int preferred, otherwise does not work well with `pack_save_dictionaries`).
        :param append_string: (optional str) string to append at the file name to `str(step)`
        :param do_print: (default as object field)
        :param collect_data: (default as object field)
        :return: the dictionary
        """
        from tensorflow import get_default_session

        if do_print is None: do_print = self.do_print
        if collect_data is None: collect_data = self.collect_data

        ss = get_default_session()
        if ss is None and do_print: print('WARNING, No default session')

        if self.timer: self.timer.stop()
        save_dict = OrderedDict([(pt[0], pt[1](step) if callable(pt[1])
                                 else ss.run(pt[1], feed_dict=pt[2](step) if callable(pt[2]) else pt[2],
                                             options=pt[3], run_metadata=pt[4]))
                                 for pt in self.processed_items])

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

    def pack_save_dictionaries(self, name='pack', append_string='', erase_others=True):
        """
        Creates an unique file starting from file created by method `save`.
        The file contains a dictionary with keys equal to save_dict keys and values list of values form original files.

        :param name:
        :param append_string:
        :param erase_others:
        :return: The generated dictionary
        """
        import glob
        all_files = sorted(glob.glob(join_paths(self.directory, FOLDER_NAMINGS['OBJ_DIR'], '*%s.pkgz' % append_string)),
                           key=os.path.getctime)  # sort by creation time
        if len(all_files) == 0:
            print('No file found')
            return

        objs = [load_obj(path, root_dir='', notebook_mode=False) for path in all_files]

        packed_dict = OrderedDict([(k, []) for k in objs[0]])
        for obj in objs:
            [packed_dict[k].append(v) for k, v in obj.items()]
        self.save_obj(packed_dict, name=name)

        if erase_others:
            [os.remove(f) for f in all_files]

        return packed_dict

    def save_fig(self, name):
        """
        Object-oriented version of `save_fig`

        :param name: name of the figure (.pdf extension automatically added)
        :return:
        """
        return save_fig(name, root_dir=self.directory,
                        default_overwrite=self.default_overwrite, notebook_mode=False)

    def save_obj(self, obj, name):
        """
         Object-oriented version of `save_obj`

        :param obj: object to save
        :param name: name of the file (.pkgz extension automatically added)
        :return:
        """
        return save_obj(obj, name, root_dir=self.directory,
                        default_overwrite=self.default_overwrite, notebook_mode=False)

    def save_adjacency_matrix_for_gephi(self, matrix, name, class_names=None):
        """
        Object-oriented version of `save_adjacency_matrix_for_gephi`

        :param matrix:
        :param name:
        :param class_names:
        :return:
        """
        return save_adjacency_matrix_for_gephi(matrix, name, root_dir=self.directory,
                                               notebook_mode=False, class_names=class_names)

    def save_setting(self, local_variables, excluded=None, append_string=''):
        """
        Object-oriented version of `save_setting`

        :param local_variables:
        :param excluded:
        :param append_string:
        :return:
        """
        return save_setting(local_variables, root_dir=self.directory, excluded=excluded,
                            default_overwrite=self.default_overwrite, collect_data=self.collect_data,
                            notebook_mode=False, do_print=self.do_print, append_string=append_string)

    def load_obj(self, name):
        """
         Object-oriented version of `load_obj`

        :param name: name of the file (.pkgz extension automatically added)
        :return: unpacked object
        """
        return load_obj(name, root_dir=self.directory, notebook_mode=False)

    # def work in progress

if __name__ == '__main__':
    sav1 = Saver('tbd',
                    'random', lambda step: np.random.randn(),
                 default_overwrite=True
                    )
    sav1.timer.start()
    sav1.save(0)
    time.sleep(2)
    sav1.save(1)
    time.sleep(1)
    sav1.save(2)
    sav1.pack_save_dictionaries()

