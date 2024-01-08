from typing import Literal, List

_BACKEND = None
PYQUDA = None


def get_backend():
    global _BACKEND
    if _BACKEND is None:
        set_backend("numpy")
    return _BACKEND


def set_backend(backend: Literal["numpy", "cupy"]):
    global _BACKEND
    if not isinstance(backend, str):
        backend = backend.__name__
    backend = backend.lower()
    assert backend in ["numpy", "cupy"]
    if backend == "numpy":
        import numpy

        _BACKEND = numpy
    elif backend == "cupy":
        import cupy

        _BACKEND = cupy
    # elif backend == "torch":
    #     import torch
    #     torch.set_default_device("cuda")
    #     _BACKEND = torch
    else:
        raise ValueError(R'backend must be "numpy", "cupy" or "torch"')


def check_QUDA(grid_size: List[int] = None):
    global PYQUDA
    if PYQUDA is None:
        try:
            # import os
            # os.environ["QUDA_RESOURCE_PATH"] = ".cache" # set your QUDA_RESOURCE_PATH before init()
            import pyquda

            pyquda.init(grid_size)
            print(pyquda.__file__)
            if pyquda.__version__ < "0.5.0":
                raise ImportError(f"PyQuda version {pyquda.__version__} < Required 0.5.0")
        except ImportError as e:
            print(f"ImportError: {e}")
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
        else:
            PYQUDA = True
    if PYQUDA is None:
        PYQUDA = False

    return PYQUDA
