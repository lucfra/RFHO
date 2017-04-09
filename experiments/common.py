from rfho.save_and_load import save_obj, load_obj
from rfho.utils import as_list, flatten_list
from rfho.experiments.greek_alphabet import greek_alphabet

import time
import threading

from functools import reduce

from collections import OrderedDict

from matplotlib import rc

rc('text', usetex=True)

GREEK_LETTERS = list(greek_alphabet.values())


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
    return {k: v.setting() if hasattr(v, 'setting') else v for k, v in local_variables.items() if v not in excluded}


def save_setting(local_variables, excluded=None, default_overwrite=False):
    dictionary = generate_setting_dict(local_variables, excluded=excluded)
    print(dictionary)
    save_obj(dictionary, 'setting', default_overwrite=default_overwrite)


def build_saver(*args):
    """
    Helper for building a saver function to collect data. Intended to be used together with OnlinePlotStream

    :param args: a list of (from pairs to at most) 5-tuples that represent the things you want to save.
                  The first arg of each tuple should be a string that will be the key of the save_dict.
                  Then there can be either a callable with signature (step) -> None
                  Should pass the various args in ths order:
                      fetches: tensor or list of tensors to compute;
                      feeds (optional): to be passed to tf.Session.run. Can be a
                      callable with signature (step) -> feed_dict
                      options (optional): to be passed to tf.Session.run
                      run_metadata (optional): to be passed to tf.Session.run
    :return: a callable with signature (tf.Session, step,  (opt) append_string, (opt, True) do_print, (opt, True)
                collect_data) -> save_dict
    """

    assert isinstance(args[0], str), 'Check args! first arg: %s. Should be a string. All args: %s' % (args[0], args)

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

    def intern(ss, step, append_string="", do_print=True, collect_data=True):
        name = str(step) + append_string
        save_dict = OrderedDict([(pt[0], pt[1](step) if callable(pt[1])
                                 else ss.run(pt[1], feed_dict=pt[2](step) if callable(pt[2]) else pt[2],
                                             options=pt[3], run_metadata=pt[4]))
                                 for pt in processed_args])

        if do_print:
            for key, v in save_dict.items(): print(key, v)
        if collect_data: save_obj(save_dict, name)

        return save_dict

    return intern


def process_name(name):
    # there might be underscores in names... split by underscores
    underscore_power_split = flatten_list([n.split('^') for n in name.split('_')])
    if any([u in GREEK_LETTERS for u in underscore_power_split]):
        return r'$\%s$' % name
    return name


def standard_plotter(ax, stream_dict, **kwargs):
    ax.clear()
    [ax.plot(v, label=k, **kwargs) for k, v in stream_dict.items()]
    if len(stream_dict) > 2:
        ax.legend(loc=4)


class OnlineReadStream:
    def __init__(self, *names):
        self.stream_dict = {n: [] for n in names}

    def process_save_dict(self, save_dict):
        [self.stream_dict[k].append(v) for k, v in save_dict.items() if k in self.stream_dict]

    def plot(self):
        pass


class OnlinePlotStream(OnlineReadStream):
    def __init__(self, ax, *names, swap_names=None, title=None, plotter=None, ax_settings=None, **kwargs):
        super().__init__(*names)
        self.ax = ax
        self.ax_settings = ax_settings or []
        self._swap_names = swap_names or {}
        self.plotter = plotter or standard_plotter
        self.kwargs = kwargs
        self.title = title

    def plot(self):
        self.plotter(self.ax, self.process_swapped_names(), **self.kwargs)
        [axs(self.ax) for axs in self.ax_settings]
        title = self.title or 'Plot of ' + ', '.join([process_name(n) for n in self.process_swapped_names()])
        self.ax.set_title(title)

    def process_swapped_names(self):
        res = {}
        for k, v in self.stream_dict.items():
            new_name = self._swap_names.get(k, k)
            res[new_name] = v
        return res

    def process_and_plot(self, save_dict):
        self.process_save_dict(save_dict)
        self.plot()


def read_stream(prefix='', start=0, stop=100000):
    res = []
    for kk in range(start, stop):
        try:
            res.append(load_obj(prefix + str(kk)))
        except FileNotFoundError:
            break
    return res


def continuous_plot(fig, plot_streams, prefix='', delay=120, additional_operations=None, start_from=0):
    plot_streams = as_list(plot_streams)
    additional_operations = additional_operations or []
    read_count = start_from
    while threading.current_thread().is_alive():
        [op.run() for op in additional_operations]
        updates = read_stream(prefix=prefix, start=read_count)
        read_count += len(updates)
        print(read_count)
        for upd in updates:
            [pls.process_save_dict(upd) for pls in plot_streams]
        [pls.plot() for pls in plot_streams]
        fig.canvas.draw()
        time.sleep(delay)


class ReadSaveDictThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, fig, plot_streams, prefix='', delay=60, additional_operations=None, start_from=0, stop_at=10000):
        super(ReadSaveDictThread, self).__init__(daemon=True)
        self._fig = fig
        self._stop = threading.Event()
        self._plot_streams = as_list(plot_streams)
        self._prefix = prefix
        self._delay = delay
        self._additional_operations = additional_operations or []
        self._start_from = start_from
        self._stop_at = stop_at
        self.read_count = start_from

    def clear(self):  # TODO implement this (should clear plot_streams)
        pass

    def stop(self):
        self._stop.set()

    def is_alive(self):
        return super(ReadSaveDictThread, self).is_alive() and not self._stop.is_set()

    def get_stream_dict(self):
        pass

    def read_data(self):
        updates = read_stream(prefix=self._prefix, start=self.read_count, stop=self._stop_at)
        self.read_count += len(updates)
        for upd in updates:
            [pls.process_save_dict(upd) for pls in self._plot_streams]

    def get_data(self):
        return reduce(lambda a, n: {**a, **n}, [pls.stream_dict for pls in self._plot_streams], {})

    def do_plots(self):
        [pls.plot() for pls in self._plot_streams]
        self._fig.canvas.draw()

    def run(self):
        while self.is_alive():
            self.read_data()
            self.do_plots()
            time.sleep(self._delay)


if __name__ == '__main__':
    print(process_name('rho_si'))
