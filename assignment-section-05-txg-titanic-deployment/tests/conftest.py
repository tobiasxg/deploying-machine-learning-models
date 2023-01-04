import logging

import pytest
from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset

# for some reason, this cannot be imported without errors
# DeprecationWarning & InvocationError
# I'll try solving it through reinstalling scipy-1.10.0
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_input_data():
    data = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],  # target
        test_size=config.model_config.test_size,  # percentage of obs in test set
        random_state=config.model_config.random_state)  # seed to ensure reproducibility

    return x_train, x_test, y_train, y_test
