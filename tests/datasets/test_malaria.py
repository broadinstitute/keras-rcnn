from __future__ import print_function
import pytest
from keras.datasets import malaria

training, test = malaria.load_data()
assert len(training) == 1208
assert len(test) == 120