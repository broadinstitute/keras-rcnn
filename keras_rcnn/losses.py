import keras.backend

if keras.backend.backend() == "tensorflow":
    import tensorflow


def smooth_l1_loss(y_true, y_pred):
    delta = 0.5

    x = keras.backend.abs(y_true - y_pred)

    p = x < delta
    q = 0.5 * x ** 2
    r = delta * (x - 0.5 * delta)

    if keras.backend.backend() == "tensorflow":
        x = tensorflow.where(p, q, r)
    else:
        x = keras.backend.switch(p, q, r)

    return keras.backend.sum(x)
