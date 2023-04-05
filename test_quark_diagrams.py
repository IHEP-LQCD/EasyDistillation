import numpy as np
import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from lattice import GaugeFieldIldg
from lattice import setBackend
# import cupy as cp

setBackend(np)

import lattice


gamma0 = lattice.gamma.gamma(0)
gamma1 = lattice.gamma.gamma(1)
gamma2 = lattice.gamma.gamma(2)
gamma3 = lattice.gamma.gamma(4)
gamma4 = lattice.gamma.gamma(8)
gamma5 = lattice.gamma.gamma(15)

print(gamma1@gamma2)
import cupy as cp
# print(gamma1@cp.asarray(gamma2))

print( cp.einsum("ij,jk->ik", gamma1, cp.asarray(gamma2)))