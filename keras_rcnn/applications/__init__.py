import os.path

import keras.utils.data_utils


def load_weights(name):
    origin = "http://keras-rcnn.storage.googleapis.com/{}.tar.gz".format(name)

    pathname = keras.utils.data_utils.get_file(
        fname=name,
        origin=origin,
        untar=True,
        cache_subdir='models'
    )

    return pathname
