import keras


class MAP(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}

        self.losses = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}

        self.losses.append(logs.get('loss'))

    @staticmethod
    def compute_mean_average_precision():
        pass
