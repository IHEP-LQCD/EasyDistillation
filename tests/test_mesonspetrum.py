import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
print(test_dir)
sys.path.insert(0, os.path.join(test_dir, ".."))

from lattice import set_backend, get_backend

set_backend("cupy")
backend = get_backend()

# from lattice.insertion.mom_dict import momDict_mom9
momentum_dict = {0: "0 0 0", 1: "0 0 1", 2: "0 1 1", 3: "1 1 1", 4: "0 0 2", 5: "0 1 2", 6: "1 1 2"}

from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName

###################        set hadron        ##################################
pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momentum_dict)
print(pi_A1[0])
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

b1xnabla_A1 = Insertion(GammaName.B1, DerivativeName.NABLA, ProjectionName.A1, momentum_dict)
print(b1xnabla_A1[0])
op_pi2 = Operator("pi2", [pi_A1[0](0, 0, 0), b1xnabla_A1[0](0, 0, 0)], [3, 1])

####################         preset      ######################################
from lattice import preset

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
Ne = 20
Ns = 4

elemental = preset.ElementalNpy(f"{test_dir}/", ".elemental.npy", [13, 6, Lt, Ne, Ne], Ne)
perambulator = preset.PerambulatorNpy(f"{test_dir}/", ".perambulators.npy", [Lt, Lt, Ns, Ns, Ne, Ne], Ne)

cfg = "weak_field"
e = elemental.load(cfg)
p = perambulator.load(cfg)
###############################################################################

from lattice.correlator.one_particle import twopoint, twopoint_matrix

# compute 2pt
twopt = twopoint([op_pi, op_pi2], e, p, list(range(Lt)), Lt)  # [Nop, Lt]
twopt = twopt.real
print(backend.arccosh((backend.roll(twopt, -1, 1) + backend.roll(twopt, 1, 1)) / twopt / 2))

# compute a 2 by 2 two-point correlation matrix
twopt_matrix = twopoint_matrix([op_pi, op_pi2], e, p, list(range(Lt)), Lt)
twopt_matrix = twopt_matrix.real
print(
    "effmass:\n",
    backend.arccosh(
        (backend.roll(twopt_matrix[0, 0], -1, 0) + backend.roll(twopt_matrix[0, 0], 1, 0)) / twopt_matrix[0, 0] / 2
    ),
)
print(
    "effmass:\n",
    backend.arccosh(
        (backend.roll(twopt_matrix[1, 1], -1, 0) + backend.roll(twopt_matrix[1, 1], 1, 0)) / twopt_matrix[1, 1] / 2
    ),
)

# compute summation of p2 = 1 2pt
# from lattice.correlator.disperion_relation import twopoint_mom2

# twopt_mom2 = twopoint_mom2(pi_A1[0], 2, e, p, list(range(Lt)), Lt)
# print(twopt_mom2)

print("Test ends!")
