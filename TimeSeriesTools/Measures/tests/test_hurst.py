
import numpy as np
from pyCausality.TimeSeries.measures import hurst


def test_Hurst():
    """Test for Hurst functions
    """
    k = [0]
    for i in range(1, 100000):
        k.append(-k[i-1]*1.0 + np.random.randn())
    k = np.array(k)
    t0 = time.time(); h = hurst(k); print time.time()-t0
    print h
    assert(h < 0.3)
    k = [0]
    for i in range(1, 100000):
        k.append(k[i-1]*1.0 + np.random.randn())
    k = np.array(k)
    t0 = time.time(); h = hurst(k); print time.time()-t0
    print h
    assert(h > 0.9)
    k = [0]
    for i in range(1, 100000):
        k.append(-k[i-1]*.0 + np.random.randn())
    k = np.array(k)
    t0 = time.time(); h = hurst(k); print time.time()-t0
    print h
    assert(np.abs(0.5-h) < 0.1)
    print 'Tests passed'


