BACKEND = None


def getNumpy():
    import numpy
    return numpy


def getBackend():
    global BACKEND
    if BACKEND is None:
        import numpy
        return numpy
    else:
        return BACKEND


def setBackend(backend):
    global BACKEND
    BACKEND = backend
