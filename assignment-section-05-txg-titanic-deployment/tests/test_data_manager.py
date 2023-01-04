
def test_loading_data(sample_input_data):
    x_train, x_test, y_train, y_test = sample_input_data
    #
    assert x_train.shape == (1178, 9)
    assert x_test.shape == (131, 9)

    # from classification_model.processing.data_manager import split_dataset
    # split_dataset(sample_input_data)
