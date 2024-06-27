# Example

In this folder, we provide usage examples for EasyDistillation.
Welcome to add more examples.

## `gen_density_peram.py`
This script generates a density perambulator, known as (VSSV) type quark loops, which are used to contract the disconnected insertion of 3pt.

## `gen_twopt.py`
This script uses the built-in function to generate a hadron 2pt function. The inclusion of disconnected diagrams (flavor singlet) are optional.

## `gen_twopt_matrix_mom.py`
This script uses the built-in function to generate a hadron 2pt function with a specific momentum square. This is typically used to calculate the dispersion relations in the meson spectrum.

## `gen_multi_draw_diagrams.py`
This script uses the input adjacency matrix to plot schematic diagrams. This example is typically used to verify the automatically generated diagram representations (adjacency matrix).

## `gen_twopt_diagram.py`
TODO: fix diagrams representaion.

## `gen_two_particle_opetators.py`
This script constructs two-particle operators with specific JPC quantum numbers. For example, it can create `PV`, `VV`, `PP` type operators.

## `gen_two_particle_corr.py`
This script contracts the two-particle (two-meson) operators into a 2pt function.

## `gen_two_particle_corr_mom.py`
This script contracts the two-particle (two-meson) operators into a 2pt function, with mesons of various relavent momenta.