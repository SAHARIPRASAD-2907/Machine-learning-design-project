from src.analysis_data import explore_data_image
from src.constants import IMG_DATA
from src.image_processing_models import (
    train_EfficientNetV2L,
    train_InceptionV3,
    train_mobile_net_v2,
    train_resnet50,
    train_VGG16,
)
from src.nlp_models import nlp_training_function
from src.pandas_function import read_data, split_data
from src.summary import summarize_image_classification, summarize_text_classification


# Function to explore bird data
def sports_data_classification():
    """
    The following function is a global function to perform all the sub operations
    1. Read the csv file
    2. Explore train data, test data and validation data
    3. Train data with ResNet50 model and generate results
    5. Train data with ResNet50 model and generate results
    6. Train data with ResNet50 model and generate results
    """
    DATA_FILE = IMG_DATA / "sports.csv"
    # Setting the directory of dataset
    # Reading the data file
    data = read_data(DATA_FILE)
    # get train_data, test_data, validate_data from the sports data
    train_data, test_data, validate_data = split_data(data)
    # Analyze Full data file
    explore_data_image(data, "Full data")
    # Analyze Train data file
    explore_data_image(train_data, "Train data")
    # Analyze Test data file
    explore_data_image(test_data, "Test data")
    # Analyze Validation data file
    explore_data_image(validate_data, "Validation data")
    # Train and generate results for Resnet 50 model
    train_resnet50()
    # Train and generate results for Mobile Net model
    train_mobile_net_v2()
    # Train and generate results for VGG Net model
    train_VGG16()
    # Train and generate results for Efficient Net model
    train_EfficientNetV2L()
    # Train and generate results for Inception Net model
    train_InceptionV3()


def pub_med_text_classification():
    # An extra function is added as there are a lot of dataset creations
    nlp_training_function()
    return 0


if __name__ == "__main__":
    sports_data_classification()
    pub_med_text_classification()
    # Summary of results
    summarize_image_classification()
    summarize_text_classification()
