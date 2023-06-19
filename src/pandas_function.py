import pandas as pd


def read_data(path: str):
    """
    The following data returns the csv file of the dataset
    """
    return pd.read_csv(path)


def split_data(data: pd.DataFrame()):
    """
    Getting the training testing and the validation data from the
    data frame
    """
    train_data = data[data["data set"] == "train"]
    test_data = data[data["data set"] == "test"].reset_index()
    valid_data = data[data["data set"] == "valid"].reset_index()
    return train_data, test_data, valid_data
