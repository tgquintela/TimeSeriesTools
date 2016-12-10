
__author__ = 'To\xc3\xb1o G. Quintela (tgq.spm@gmail.com)'
__version__ = '0.0.0'

#from pyCausality.TimeSeries.TS import *


#from pyCausality.TimeSeries.automatic_thresholding import *
#from pyCausality.TimeSeries.distances import *
#from pyCausality.TimeSeries.measures import *
#from pyCausality.TimeSeries.smoothing import *
#from pyCausality.TimeSeries.transformations import *
from tests import test_artificial_data
from tests import test_utils
from tests import test_measures
from tests import test_transformations
from tests import test_burstdetection
from tests import test_tsstatistics
from tests import test_regimedetection
from tests import test_feature_extraction
from tests import test_similarities

## Administrative information
import release
import version

## Not inform about warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
warnings.simplefilter("ignore")


def test():
    ## Tests of modules
#    test_artificial_data.test()
##    test_utils.test()
##    test_measures.test()
#    test_transformations.test()
    test_burstdetection.test()
    test_tsstatistics.test()
    test_regimedetection.test()
    test_feature_extraction.test()
##    test_similarities.test()
