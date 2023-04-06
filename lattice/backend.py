BACKEND = None
PYQUDA = None


def getNumpy():
    import numpy
    return numpy


def getBackend():
    global BACKEND
    if BACKEND is None:
        import numpy
        BACKEND = numpy
    return BACKEND


def setBackend(backend):
    global BACKEND
    BACKEND = backend


def checkQUDA():
    global PYQUDA
    if PYQUDA is None:
        try:
            import os
            from pyquda import mpi
            os.environ["QUDA_RESOURCE_PATH"] = ".cache"
            mpi.init()
        except ImportError:
            pass
        except RuntimeError:
            pass
        else:
            PYQUDA = True
    else:
        PYQUDA = False

    return PYQUDA
