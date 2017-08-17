import theano.tensor


# TODO: emulate NumPy semantics
def argsort(a):
    return theano.tensor.argsort(a)
