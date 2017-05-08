import threading
import time
from functools import reduce

from matplotlib import rc

from rfho.examples.greek_alphabet import greek_alphabet
from rfho.save_and_load import load_obj
from rfho.utils import as_list, flatten_list, merge_dicts

from rfho.save_and_load import Saver

rc('text', usetex=True)

GREEK_LETTERS = list(greek_alphabet.values())


def process_name(name):
    # there might be underscores in names... split by underscores
    underscore_power_split = flatten_list([n.split('^') for n in name.split('_')])
    if any([u in GREEK_LETTERS for u in underscore_power_split]):
        return r'$\%s$' % name
    return name


def standard_plotter(**plot_kwargs):
    """
    This is a standard plotter that would do for most of the occasions, when there are stream dict are
    matrices of values that represents scalar measures. If the relative stream_dict has more than one key
    then standard_plotter will automatically add a legend.

    :return: A callable, internally called by instances of OnlinePlotStream
    """

    def intern(ax, stream_dict, **kwargs):

        ax.clear()
        [ax.plot(v, label=k, **merge_dicts(kwargs, plot_kwargs)) for k, v in stream_dict.items()]
        if len(stream_dict) > 2:
            ax.legend(loc=0)
    return intern


def scalar_value_gradient_plotter(value_color, gradient_color, value_kwargs=None, grad_kwargs=None):
    """
    This is a plotter thought for stream_dict that contains hyperparameters values and hyper-gradients.
    Will plot these two scalar sequences on different scales.

    :param grad_kwargs:
    :param value_kwargs:
    :param value_color:
    :param gradient_color:
    :return:
    """
    value_kwargs = value_kwargs or {}
    grad_kwargs = grad_kwargs or {}

    ax2 = None

    def intern(ax, stream_dict, **kwargs):
        nonlocal ax2

        ax.clear()
        if ax2 is None:
            ax2 = ax.twinx()
        ax2.clear()

        data = list(stream_dict.values())[0]
        val = [c[0] for c in data]
        grad = [c[1] for c in data]

        val_line, = ax.plot(val, color=value_color, **merge_dicts(kwargs, value_kwargs))
        grad_line, = ax2.plot(grad, color=gradient_color, **merge_dicts(kwargs, grad_kwargs))
        ax.legend((val_line, grad_line), ('Value', 'Gradient'), loc=0)
    return intern


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
        self.plotter = plotter or standard_plotter()
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

        self.exc = []

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
        import sys
        # noinspection PyBroadException
        try:
            while self.is_alive():
                self.read_data()
                self.do_plots()
                time.sleep(self._delay)
        except Exception:
            self.exc.append(sys.exc_info())

