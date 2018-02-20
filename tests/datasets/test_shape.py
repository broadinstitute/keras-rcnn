def test_load_data(training_dictionary, test_dictionary):
    assert len(training_dictionary) == 256

    assert len(test_dictionary) == 256
