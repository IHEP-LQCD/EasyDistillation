# EasyDistillation

EasyDistillation is a `Python` package that integrates the procedure of distillation method as proposed first by ref.[X].
This package makes up of the generation of laplacian eigenvectors, perambulators, elementals, and automatic contraction, with multi-backend support both on CPU / GPU.
It features a flexible and efficient designed for optimal performance with the `Generator` class.
This package is desigened to simplify the complexities associated with the distillation process, make it accessible and efficient computationally for physical calculations using 
the distillation method.

## File IO
Using `latticce.preset` to initialize metadata of input file. For example, file data `GaugeField` / `Eigenvector` /  `Elemental` / `Perambulator`. and file type `BinaryFile` / `IldgFile`, numpy `NdarrayFile`, etc.

## Generate Laplacian eigenvectors

Refer to `EigenvectorGenerator` in `tests\test_eigenvector.py`, the Laplacian eigensolover support either Python LinearOperator eigensolver with `scipy` on cpu / `cupyx.sparse` on GPU, or [PyQuda](https://github.com/IHEP-LQCD/PyQuda) + `QUDA` library. The gauge smeaing also depend on  [PyQuda](https://github.com/IHEP-LQCD/PyQuda).

## Elemental

Refer to class `ElementalGenerator` in `tests\test_elemental.py`,  as input eigenvector data, It generate up to n-deriv ($n<3$) elemental data sequentally.

## Perambulator

Refer to class `PerambulatorGenerator` in `tests/test_perambulator_mpi.py` with  [PyQuda](https://github.com/IHEP-LQCD/PyQuda).

## Contraction
Refer to `example/gen_two_particle_corr_mom.py`.

