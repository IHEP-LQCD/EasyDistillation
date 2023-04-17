from .backend import get_backend, set_backend, check_QUDA
from .dispatch import Dispatch, processbar
from .insertion import gamma, derivative, mom_dict
from .preset import (
    GaugeFieldTimeSlice, EigenVectorTimeSlice, PerambulatorBinary, GaugeFieldIldg, ElementalNpy, EigenVectorNpy,
    Jpsi2gammaNpy
)
from .quark_diagram import (
    QuarkDiagram, Meson, Propagator, PropagatorLocal, compute_diagrams, compute_diagrams_multitime
)

from .timer import mytimer
# from .multiquarks import *
