

"""This module encapsulate the functions related with the spike detection task
and the main class to perform it.
"""


class Spike_Detection():
    """Class implemented to act as a wrapper for all the methods of spike
    detection implemented in the package.
    """

    def _init__(self, method, filepath=None):
        self.method = method
        if method == 'file':
            self.filepath = filepath
        elif method == 'default':
            pass
        elif method == 'terminal':
            self.spk_dt = Sorting_detection()
        elif method == 'gui':
            pass

    def detect(self, activation):
        dist_ts, ys = self.spk_dt.detec(activation)
        return dist_ts
