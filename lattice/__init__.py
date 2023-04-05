_QUDA_avaliable = False
try:
    import os
    from pyquda import mpi
    os.environ["QUDA_RESOURCE_PATH"] = ".cache"
    mpi.init()
except ImportError:
    pass
except RuntimeError:
    pass
else:
    _QUDA_avaliable = True


def QUDA():
    return _QUDA_avaliable


from .backend import getBackend, setBackend
from .dispatch import Dispatch, processBar
from .elemental import ElementalGenerator
from .preset import GaugeFieldTimeSlice, EigenVectorTimeSlice, PerambulatorBinary, GaugeFieldIldg, ElementalNpy, Jpsi2gammaNpy
from .insertion import gamma, derivative, mom_dict, deriv_dict

from .timer import mytimer
# from .multiquarks import *
