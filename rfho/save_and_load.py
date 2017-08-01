import time
from collections import OrderedDict, defaultdict
from functools import reduce, wraps
from inspect import signature

import matplotlib.pyplot as plt
import tensorflow as tf

import rfho as rf
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

try:
    from tabulate import tabulate
except ImportError:
    print('Might want to install library "tabulate" for a better dictionary printing')
    tabulate = None


def join_paths(*paths):
    return reduce(lambda acc, new_path: os.path.join(acc, new_path), paths)


SAVE_SETTINGS = {
    'NOTEBOOK_TITLE': ''
}

_EXP_ROOT_FOLDER = os.getenv('RFHO_EXP_FOLDER')
if _EXP_ROOT_FOLDER is None:
    print('Environment variable RFHO_EXP_FOLDER not found. Current directory will be used')
    _EXP_ROOT_FOLDER = join_paths(os.getcwd(), 'Experiments')
print('Experiment save directory is ', _EXP_ROOT_FOLDER)

FOLDER_NAMINGS = {  # TODO should go into a settings file?
    'EXP_ROOT': _EXP_ROOT_FOLDER,
    'OBJ_DIR': 'Obj_data',
    'PLOTS_DIR': 'Plots',
    'MODELS_DIR': 'Models',
    'GEPHI_DIR': 'GePhi',
}


def check_or_create_dir(directory, notebook_mode=True, create=True):
    if not os.path.exists(directory) and create:
        os.mkdir(directory)
        print('folder', directory, 'has been created')

    if notebook_mode and SAVE_SETTINGS['NOTEBOOK_TITLE']:
        directory = join_paths(directory, SAVE_SETTINGS['NOTEBOOK_TITLE'])  # += '/' + settings['NOTEBOOK_TITLE']
        if not os.path.exists(directory) and create:
            os.mkdir(directory)
            print('folder ', directory, 'has been created')
    return directory


def save_fig(name, root_dir=None, notebook_mode=True, default_overwrite=False, extension='pdf', **savefig_kwargs):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['PLOTS_DIR']),
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory, '%s.%s' % (name, extension))  # directory + '/%s.pdf' % name
    if not default_overwrite and os.path.isfile(filename):
        # if IPython is not None:
        #     IPython.display.display(tuple(IFrame(filename, width=800, height=600)))  # FIXME not working!
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
    plt.savefig(filename, **savefig_kwargs)
    # print('file saved')


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
        # print('File saved!')

def save_text(text, name, root_dir=None, notebook_mode=True, default_overwrite=False):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir),
                                    notebook_mode=notebook_mode)

    filename = join_paths(directory, '%s.txt' % name)  # directory + '/%s.pkgz' % name
    if not default_overwrite and os.path.isfile(filename):
        overwrite = input('A file named %s already exists. Overwrite (Leave string empty for NO!)?' % filename)
        if not overwrite:
            print('No changes done.')
            return
        print('Overwriting...')

    with open(filename, "w") as text_file:
        text_file.write(text)


def load_obj(name, root_dir=None, notebook_mode=True):
    if root_dir is None: root_dir = os.getcwd()
    directory = check_or_create_dir(join_paths(root_dir, FOLDER_NAMINGS['OBJ_DIR']),
                                    notebook_mode=notebook_mode, create=False)

    filename = join_paths(directory, name if name.endswith('.pkgz') else name + '.pkgz')
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
        if tabulate:
            print(tabulate(dictionary.items(), headers=('settings var names', 'values')))
        else:
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
    """
    Stopwatch class for timing the experiments. Uses `time` module.
    """

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

    def reset(self):
        self._starting_times = []
        self._stopping_times = []
        self._running = False

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
        res = sum(self.raw_elapsed_time_list()) / Timer._div_unit[self.unit]
        return res if not self.round_off else int(res)


class Saver:
    """
    Class for recording experiment data
    """

    SKIP = 'SKIP'  # skip saving value in save_dict

    def __init__(self, experiment_names, *items, append_date_to_name=True,
                 root_directory=FOLDER_NAMINGS['EXP_ROOT'],
                 timer=None, do_print=True, collect_data=True, default_overwrite=False):
        """
        Initialize a saver to collect data. (Intended to be used together with OnlinePlotStream.)

        :param experiment_names: string or list of strings which represent the name of the folder (and sub-folders)
                                    experiment oand
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
        experiment_names = as_list(experiment_names)
        if append_date_to_name:
            from datetime import datetime
            experiment_names += [datetime.today().strftime('%d-%m-%y__%Hh%Mm')]
        self.experiment_names = list(experiment_names)

        if not os.path.isabs(experiment_names[0]):
            self.directory = join_paths(root_directory)  # otherwise assume no use of root_directory
            if collect_data:
                check_or_create_dir(root_directory, notebook_mode=False)
        else:
            self.directory = ''
        for name in self.experiment_names:
            self.directory = join_paths(self.directory, name)
            check_or_create_dir(self.directory, notebook_mode=False)

        self.do_print = do_print
        self.collect_data = collect_data
        self.default_overwrite = default_overwrite

        assert isinstance(timer, Timer) or timer is None or timer is False, 'timer param not good...'

        if timer is None:
            timer = Timer()

        self.timer = timer

        self.clear_items()

        self.add_items(*items)

    # noinspection PyAttributeOutsideInit
    def clear_items(self):
        """
        Removes all previously inserted items
        
        :return: 
        """
        self._processed_items = []
        self._step = -1

    @staticmethod
    def process_items(*items):
        """
        Add items to the save dictionary

        :param items: a list of (from pairs to at most) 5-tuples that represent the things you want to save.
                      The first arg of each tuple should be a string that will be the key of the save_dict.
                      Then there can be either a callable with signature (step) -> result or () -> result
                      or tensorflow things... In this second case you  should pass the following args in ths order:
                          fetches: tensor or list of tensors to compute;
                          feeds (optional): to be passed to tf.Session.run. Can be a
                                            callable with signature (step) -> feed_dict
                                            or () -> feed_dict
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
            if callable(part[1]):  # python stuff
                if len(part) == 2: part.append(True)  # always true default condition
            else:  # tensorflow stuff
                part += [None] * (6 - len(part))  # representing name, fetches, feeds, options, metadata
                if part[3] is None: part[3] = True  # default condition
            processed_args.append(part)
        # self._processed_items += processed_args
        # return [pt[0] for pt in processed_args]
        return processed_args

    def add_items(self, *items):
        """
        Adds internally items to this saver

        :param items:
        :return:
        """
        processed_items = Saver.process_items(*items)
        self._processed_items += processed_items
        return [pt[0] for pt in processed_items]

    def save(self, step=None, session=None, append_string="", do_print=None, collect_data=None,
             processed_items=None, _res=None):
        """
        Builds and save a dictionary with the keys and values specified at construction time or by method
        `add_items`

        :param processed_items: optional, processed item list (returned by add_items)
                                if None uses internally stored items
        :param session: Optional tensorflow session, otherwise uses default session
        :param step: optional step, if None (default) uses internal step
                        (int preferred, otherwise does not work well with `pack_save_dictionaries`).
        :param append_string: (optional str) string to append at the file name to `str(step)`
        :param do_print: (default as object field)
        :param collect_data: (default as object field)
        :param _res: used internally by context managers

        :return: the dictionary
        """
        from tensorflow import get_default_session
        if step is None:
            self._step += 1
            step = self._step
        if not processed_items: processed_items = self._processed_items

        if do_print is None: do_print = self.do_print
        if collect_data is None: collect_data = self.collect_data

        if session:
            ss = session
        else:
            ss = get_default_session()
        if ss is None and do_print: print('WARNING, No tensorflow session available')

        if self.timer: self.timer.stop()

        def _maybe_call(_method):
            if not callable(_method): return _method
            if len(signature(_method).parameters) == 0:
                return _method()
            elif len(signature(_method).parameters) == 1:
                return _method(step)
            else:  # complete signature?
                return _method(step, _res)

        save_dict = OrderedDict([(pt[0], _maybe_call(pt[1]) if callable(pt[1])
        else ss.run(pt[1], feed_dict=_maybe_call(pt[2]),
                    options=pt[4], run_metadata=pt[5])
        if _maybe_call(pt[2 if callable(pt[1]) else 3]) else Saver.SKIP)
                                 for pt in processed_items]
                                )

        if self.timer: save_dict['Elapsed time (%s)' % self.timer.unit] = self.timer.elapsed_time()

        if do_print:
            if tabulate:
                print(tabulate(save_dict.items(), headers=('Step %s' % step, 'Values'), floatfmt='.5f'))
            else:
                print('SAVE DICT:')
                for key, v in save_dict.items():
                    print(key, v, sep=': ')
                print()
        if collect_data:
            self.save_obj(save_dict, str(step) + append_string)

        if self.timer: self.timer.start()

        return save_dict

    def pack_save_dictionaries(self, name='all', append_string='', erase_others=True):
        """
        Creates an unique file starting from file created by method `save`.
        The file contains a dictionary with keys equal to save_dict keys and values list of values form original files.

        :param name:
        :param append_string:
        :param erase_others:
        :return: The generated dictionary
        """
        import glob
        all_files = sorted(glob.glob(join_paths(
            self.directory, FOLDER_NAMINGS['OBJ_DIR'], '[0-9]*%s.pkgz' % append_string)),
            key=os.path.getctime)  # sort by creation time
        if len(all_files) == 0:
            print('No file found')
            return

        objs = [load_obj(path, root_dir='', notebook_mode=False) for path in all_files]

        # packed_dict = OrderedDict([(k, []) for k in objs[0]])

        # noinspection PyArgumentList
        packed_dict = defaultdict(list, OrderedDict())
        for obj in objs:
            [packed_dict[k].append(v) for k, v in obj.items()]
        self.save_obj(packed_dict, name=name + append_string)

        if erase_others:
            [os.remove(f) for f in all_files]

        return packed_dict

    def record(self, *what, append_string=''):  # TODO  this is un initial (maybe bad) idea.
        """
        Context manager for saver. saves executions

        :param what:
        :param append_string:
        :return:
        """

        return Records.on_hyperiteration(self, *what, append_string=append_string)  # FIXME to be finished

    def save_text(self, text, name):

        return save_text(text=text, name=name, root_dir=self.directory, default_overwrite=self.default_overwrite,
                         notebook_mode=False)

    def save_fig(self, name, extension='pdf', **savefig_kwargs):
        """
        Object-oriented version of `save_fig`

        :param extension: 
        :param name: name of the figure (.pdf extension automatically added)
        :return:
        """
        return save_fig(name, root_dir=self.directory, extension=extension,
                        default_overwrite=self.default_overwrite, notebook_mode=False,
                        **savefig_kwargs)

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
        excluded = as_list(excluded or [])
        excluded.append(self)  # no reason to save itself...
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


# noinspection PyPep8Naming
def Loader(folder_name):
    """
    utility method for creating a Saver with loading intentions,
    does not create timer nor append time to name. just give the folder name
    for the saver

    :param folder_name: (string or list of strings)
                        either absolute or relative, in which case root_directory will be used
    :return: a `Saver` object
    """
    return Saver(folder_name, append_date_to_name=False, timer=False,
                 collect_data=False)


class Records:
    """
    Contains (for the moment) static convenience methods for recording quantities
    """

    class on_hyperiteration:
        """
        context for record at each hyperiteration
        """

        def __init__(self, saver, *record_what, append_string='', do_print=None, collect_data=None):
            self.saver = saver
            self.append_string = append_string
            if self.append_string: self.append_string = '__' + self.append_string
            self.do_print = do_print
            self.collect_data = collect_data

            self._unwrapped = []

            self._record_what = record_what or []
            self._processed_items = []

            self._step = 0

        def __enter__(self):
            self._wrap()

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_tb:
                self.saver.save_obj((str(exc_type), str(exc_val), str(exc_tb)),
                                    'exception' + self.append_string)
            self.saver.pack_save_dictionaries(append_string=self.append_string)

            self._unwrap()

            # TODO is this a good thing? or should we leave it to do manually
            self.saver.clear_items()
            if self.saver.timer: self.saver.timer.stop()

        def _wrap(self):
            self._unwrapped.append(rf.HyperOptimizer.initialize)
            rf.HyperOptimizer.initialize = self._initialize_wrapper(rf.HyperOptimizer.initialize)

            self._unwrapped.append(rf.HyperOptimizer.run)
            rf.HyperOptimizer.run = self._saver_wrapper(rf.HyperOptimizer.run)  # mmm...

        def _unwrap(self):
            rf.HyperOptimizer.initialize = self._unwrapped[0]
            rf.HyperOptimizer.run = self._unwrapped[1]

        def _saver_wrapper(self, f):
            @wraps(f)
            def _saver_wrapped(*args, **kwargs):
                res = f(*args, **kwargs)
                self._execute_save(res, *args, **kwargs)
                return res

            return _saver_wrapped

        def _initialize_wrapper(self, f):  # this should be good since
            @wraps(f)
            def _initialize_wrapped(*args, **kwargs):
                first_init = f(*args, **kwargs)
                # add savers just at the first initialization
                if first_init:
                    self._processed_items += rf.flatten_list(
                        [Saver.process_items(*e(*args, **kwargs)) for e in self._record_what])
                    self._execute_save('INIT', *args, **kwargs)

                return first_init

            return _initialize_wrapped

        # noinspection PyUnusedLocal
        def _execute_save(self, res, *args, **kwargs):  # maybe args and kwargs could be useful...
            self.saver.save(step=self._step, append_string=self.append_string,
                            processed_items=self._processed_items,
                            do_print=self.do_print, collect_data=self.collect_data,
                            _res=res)
            self._step += 1

    # noinspection PyClassHasNoInit,PyPep8Naming
    class on_forward(on_hyperiteration):  # context class
        """
        Saves at every iteration (before call of method `step_forward`)
        """

        def _wrap(self):
            self._unwrapped.append(rf.HyperOptimizer.initialize)
            rf.HyperOptimizer.initialize = self._initialize_wrapper(rf.HyperOptimizer.initialize)

            self._unwrapped.append(rf.ForwardHG.step_forward)
            rf.ForwardHG.step_forward = self._saver_wrapper(rf.ForwardHG.step_forward)  # mmm...

        def _unwrap(self):
            rf.HyperOptimizer.initialize = self._unwrapped[0]
            rf.ForwardHG.step_forward = self._unwrapped[1]

    @staticmethod
    def direct(*items):
        """
        Everything passed in items is passed directly to `Saver.

        :param items:
        :return:
        """

        # noinspection PyUnusedLocal
        def _call(*args, **kwargs):
            return items

        return _call

    @staticmethod
    def norms_of_z():
        """

        :return:
        """

        def _call(*args, **kwargs):
            hg = args[0]
            if isinstance(hg, rf.HyperOptimizer): hg = hg.hyper_gradients  # guess most common case
            assert isinstance(hg, rf.ForwardHG)
            _rs = Records.tensors(*hg.zs, op=tf.norm)(args, kwargs)
            return _rs

        return _call

    @staticmethod
    def norms_of_d_dynamics_d_hypers(fd=None):
        """
        In `ForwardHG` records the norm of the partial derivatives of the dynamics w.r.t. the hyperparameters.

        :param fd:
        :return:
        """
        if fd is None: fd = lambda stp, rs: rs

        def _call(*args, **kwargs):
            hg = args[0]
            if isinstance(hg, rf.HyperOptimizer):
                hg = hg.hyper_gradients  # guess most common case
            assert isinstance(hg, rf.ForwardHG)
            _rs = Records.tensors(*hg.d_dynamics_d_hypers, op=tf.norm,
                                  fd=fd,
                                  condition=lambda stp, rs: rs != 'INIT')(args, kwargs)
            return _rs

        return _call

    @staticmethod
    def hyperparameters():
        """
        Simple one! record all hyperparameter values, assuming the usage of `HyperOptimizer`

        :return: a function
        """

        # noinspection PyUnusedLocal
        def _call(*args, **kwargs):
            hyper_optimizer = args[0]
            assert isinstance(hyper_optimizer, rf.HyperOptimizer)
            return rf.flatten_list(
                [rf.simple_name(hyp), hyp]
                for hyp in hyper_optimizer.hyper_list)

        return _call

    @staticmethod
    def hypergradients():
        """
        Record all hypergradient values, assuming the usage of `HyperOptimizer`

        :return:
        """

        # noinspection PyUnusedLocal
        def _call(*args, **kwargs):
            hyper_optimizer = args[0]
            assert isinstance(hyper_optimizer, rf.HyperOptimizer)
            return rf.flatten_list(
                ['grad::' + rf.simple_name(hyp), hyper_optimizer.hyper_gradients.hyper_gradients_dict[hyp]]
                for hyp in hyper_optimizer.hyper_list)

        return _call

    @staticmethod
    def tensors(*tensors, key=None, scope=None, name_contains=None,
                rec_name='', op=tf.identity, fd=None,
                condition=True):
        """
        Little more difficult... attempts to record tensor named name

        :param name_contains: record all tensors which name contains this string. Can be a list.
        :type condition: bool | function
        :param condition: optional condition for triggering the saving of tensors, can have different
                            signatures
        :param tensors: varargs of tensor names
        :param scope: optional for collections
        :param key: to record collections
        :param op: optional operation to apply to each tensor
        :param rec_name: optional name to prepend to all tensors recorded by this
        :param fd: # given to _process_feed_dicts_for_rec
        :return:
        """
        if rec_name: rec_name += '::'  # maybe find a better way

        def _call(*args, **_kwargs):
            if tensors:
                _tensors = [tf.get_default_graph().get_tensor_by_name(tns + ':0') if isinstance(tns, str)
                            else tns for tns in tensors]
            elif key:
                _tensors = tf.get_collection(key, scope=scope)
            elif name_contains:
                _names = rf.flatten_list([[n.name for n in tf.get_default_graph().as_graph_def().node
                                           if nc in n.name] for nc in as_list(name_contains)])
                return Records.tensors(*_names, rec_name=rec_name, op=op, fd=fd, condition=True)(*args, **_kwargs)
            else:
                raise NotImplemented('One between key and names should be given')
            # try with dictionary of form (string (simple name of placeholder), data)
            _rs2 = rf.flatten_list([rec_name + rf.simple_name(tns.name),
                                    op(tns),
                                    Records._process_feed_dicts_for_rec(fd, *args, **_kwargs),
                                    condition]
                                   for tns in _tensors)
            return _rs2

        return _call

    @staticmethod
    def model():  # TODO discuss with others to see what's best way to save models...
        """
        Should save the model(s) in a useful way..

        :return:
        """
        raise NotImplemented()

    @staticmethod
    def setting():  # TODO I have no precise idea on how to implement this...
        """
        Should save experiment meta-info like params, dataset, beginning/end...
        name of experiment function, git version and so on.

        :return:
        """
        raise NotImplemented()

    @staticmethod
    def _process_feed_dicts_for_rec(fd, *args, **kwargs):
        # TODO add more functionality...
        """

        # try with dictionary of form (string (simple name of placeholder), data)

        :param fd:
        :param args:  # might be useful??
        :param kwargs:
        :return:
        """
        if fd is None or callable(fd): return fd

        def _std_process_dict(_dict):
            return {tf.get_default_graph().get_tensor_by_name(n + ':0'): v for n, v in _dict.items()}

        def _fds():
            if isinstance(fd, dict):
                _rs = _std_process_dict(fd)

            elif isinstance(fd, (list, tuple)):  # (x, y, dataset)
                if len(fd) == 3 and isinstance(fd[2], rf.Dataset):  # very common scenario
                    _rs = {tf.get_default_graph().get_tensor_by_name(fd[0] + ':0'): fd[2].data,
                           tf.get_default_graph().get_tensor_by_name(fd[1] + ':0'): fd[2].target,
                           }
                else:
                    raise NotImplemented('not understood')
            else:
                raise NotImplemented('not understood')

            return _rs

        return _fds


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
