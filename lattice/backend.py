BACKEND = None


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
