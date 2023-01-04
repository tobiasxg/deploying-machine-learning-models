import re

from typing import List

# to handle datasets
import pandas as pd
import numpy as np

from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


# retain only the first cabin if more than
# 1 are available per passenger
def get_first_cabin(row):
    try:
        return row.split()[0]
    except AttributeError:
        return np.nan


# extracts the title (Mr, Ms, etc) from the name variable
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'


def pre_pipeline_data_processing(input_data: pd.DataFrame) -> pd.DataFrame:
    # replace interrogation marks by NaN values
    data = input_data.replace('?', np.nan)

    # retain only the first cabin if more than
    # 1 are available per passenger
    data['cabin'] = data['cabin'].apply(get_first_cabin)

    # extracts the title (Mr, Ms, etc) from the name variable
    data['title'] = data['name'].apply(get_title)

    # cast numerical variables as floats
    data['fare'] = data['fare'].astype('float')
    data['age'] = data['age'].astype('float')

    # drop unnecessary variables
    data.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data


# load dataset
def load_dataset() -> pd.DataFrame:
    input_data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = pre_pipeline_data_processing(input_data)
    return data


def split_dataset(data) -> List[pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    return x_train, x_test, y_train, y_test
