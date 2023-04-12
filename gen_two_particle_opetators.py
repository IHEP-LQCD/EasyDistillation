# examples for constructing two-particle operators
from lattice.symmetry.two_particle import two_particle_Cartesian_basis

print("S-wave, PV type opetator with mom=2")
for i in two_particle_Cartesian_basis(op1="V", op2="P", J=1, L=0, Spin=1, mom2=2):
    print(i, end="\n\n")

print("P-wave PV type opetator with mom=1")
for i in two_particle_Cartesian_basis(op1="P", op2="V", J=1, L=1, Spin=1, mom2=1):
    print(i, end="\n\n")

print("P-wave VV type opetator with mom=1")
for i in two_particle_Cartesian_basis(op1="V1", op2="V2", J=1, L=1, Spin=1, mom2=1):
    print(i, end="\n\n")

# TODO:add API for Operator class
