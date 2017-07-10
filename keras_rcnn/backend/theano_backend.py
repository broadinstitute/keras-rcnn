import theano.tensor


def argsort(a, axis=1):
    return theano.tensor.argsort(a, axis)
