# to divide train and test set
from sklearn.model_selection import train_test_split

from config.core import config

from processing.data_manager import load_dataset


def train() -> None:

    data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.config_model.features],  # predictors
        data[config.config_model.target],  # target
        test_size=config.config_model.test_size,  # percentage of obs in test set
        random_state=config.config_model.random_state)  # seed to ensure reproducibility

