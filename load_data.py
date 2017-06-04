from __future__ import absolute_import

import pickle


def load_data():
    """Loads malaria dataset.
    # Returns
        Tuple of dictionaries: `train, validation, test`.
    """
    return pickle.load('/home/jhung0/malaria_data.pkl')
