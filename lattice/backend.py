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


def check_QUDA(grid_size: List[int] = None, backend: Literal["cupy", "torch"] = "cupy", resource_path: str = None):
    global PYQUDA
    if PYQUDA is None:
        try:
            import pyquda

            pyquda.init(grid_size, backend=backend, resource_path=resource_path)
            print("PyQUDA installed in: ", pyquda.__file__)
            from packaging.version import Version

            if Version(pyquda.__version__) < Version("0.9.0"):
                raise ImportError(f"PyQuda version {pyquda.__version__} < Required 0.9.X")

        except ImportError as e:
            print(f"ImportError: {e}")
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
        else:
            PYQUDA = True
    if PYQUDA is None:
        PYQUDA = False

    return PYQUDA
