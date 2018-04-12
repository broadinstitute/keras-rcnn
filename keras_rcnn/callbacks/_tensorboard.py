import keras.callbacks


class TensorBoard(keras.callbacks.Callback):
    def __init__(self):
        super(TensorBoard, self).__init__()
