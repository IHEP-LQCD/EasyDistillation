BACKEND = None
PYQUDA = None


def get_numpy():
    import numpy
    return numpy


def get_backend():
    global BACKEND
    if BACKEND is None:
        import numpy
        BACKEND = numpy
    return BACKEND


def set_backend(backend):
    global BACKEND
    BACKEND = backend


def check_QUDA():
    global PYQUDA
    if PYQUDA is None:
        try:
            import os
            from pyquda import mpi
            os.environ["QUDA_RESOURCE_PATH"] = ".cache"
            mpi.init()
        except ImportError as e:
            print(F"ImportError: {e}")
        except RuntimeError as e:
            print(F"RuntimeError: {e}")
        else:
            PYQUDA = True
    else:
        PYQUDA = False

    return PYQUDA
