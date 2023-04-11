from .backend import getBackend, setBackend, checkQUDA
from .dispatch import Dispatch, processBar
from .insertion import gamma, derivative, mom_dict
from .preset import (
    GaugeFieldTimeSlice, EigenVectorTimeSlice, PerambulatorBinary, GaugeFieldIldg, ElementalNpy, EigenVectorNpy,
    Jpsi2gammaNpy
)

from .timer import mytimer
# from .multiquarks import *
