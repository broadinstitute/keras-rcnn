from .common import *

if keras.backend.backend() == "theano":
    from .theano_backend import *
elif keras.backend.backend() == "tensorflow":
    from .tensorflow_backend import *
else:
    pass
