import threading

import numpy
import numpy.random
import skimage.io
import keras.utils
import sklearn.preprocessing


class Iterator:
    def __init__(self, n, batch_size, shuffle, seed):
        self.batch_index = 0

        self.batch_size = batch_size

        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

        self.lock = threading.Lock()

        self.n = n

        self.shuffle = shuffle

        self.total_batches_seen = 0

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()

        while True:
            if seed is not None:
                numpy.random.seed(seed + self.total_batches_seen)

            if self.batch_index == 0:
                index_array = numpy.arange(n)

                if shuffle:
                    index_array = numpy.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n

            if n > current_index + batch_size:
                current_batch_size = batch_size

                self.batch_index += 1
            else:
                current_batch_size = n - current_index

                self.batch_index = 0

            self.total_batches_seen += 1

            yield index_array[current_index:current_index + current_batch_size], current_index, current_batch_size

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self, *args, **kwargs):
        pass


class DictionaryIterator(Iterator):
    def __init__(self, dictionary, generator, shuffle=False, seed=None):
        self.dictionary = dictionary

        self.encoder = sklearn.preprocessing.LabelEncoder()

        self.generator = generator

        self.encoder.fit(generator.classes)

        Iterator.__init__(self, len(dictionary), 1, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        index = index_array[0]

        pathname = self.dictionary[index]["filename"]

        image = skimage.io.imread(pathname)

        image = numpy.expand_dims(image, 0)

        ds = self.dictionary[index]["boxes"]

        boxes = numpy.asarray([[d[k] for k in ['y1', 'x1', 'y2', 'x2']] for d in ds])

        labels = numpy.asarray([d['class'] for d in ds])

        labels = self.encoder.transform(labels)

        labels = keras.utils.to_categorical(labels)

        metadata = list(image.shape[1:-1]) + [1]

        # boxes = numpy.expand_dims(boxes, 0)

        # labels = numpy.expand_dims(labels, 0)

        # metadata = numpy.expand_dims(metadata, 0)

        return [image, boxes, labels, metadata], [boxes, labels]


class ObjectDetectionGenerator:
    def __init__(self, classes):
        self.classes = classes

    def flow(self, dictionary, shuffle=True, seed=None):
        return DictionaryIterator(dictionary, self, shuffle=shuffle, seed=seed)
