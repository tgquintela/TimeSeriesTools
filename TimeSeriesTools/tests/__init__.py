


#def sampling_ts_randompoints(ts, values, t_lim, t_inter):
#    np.random.permutation()
#    times = create_times_randompoints(t_lim, t_inter)
#    ts_sampling = ts[times]
#    values_sampling = values[times]
#    return ts_sampling
    

#class TimeSeries(object):
#    """Main object which wrappes all the time series information to be easy to
#    move and pass through the framework.
#
#    """
#
#    def _initialization(self):
#        self._set = False
#        self.regular = None
#        self.times = None
#        self.elements = None
#
#    def __init__(self, values, elements=None, times=None, regular=None):
#        self.set_values(values)
#        self._set_elements(elements)
#
#    def set_values(self, values):
#        if not self._set:
#            assert(type(values) == np.ndarray)
#            if len(values.shape) != 3:
#                values = np.atleast_3d(values)
#            assert(len(values.shape) == 3)
#        else:
#            n_elements = len(self.elements)
#
#    def _set_elements(self, elements):
#        pass
