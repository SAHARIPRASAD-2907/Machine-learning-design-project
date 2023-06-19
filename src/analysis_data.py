import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from src.constants import FIGS, OUTPUTS, IMG_DATA

"""
The following function is written based in the techniques in the following blog
https://medium.com/analytics-vidhya/introduction-to-exploratory-data-analysis-for-image-text-based-data-1179e194df3f
"""


def get_shape_size_and_info(data: pd.DataFrame(), frame_type: str):
    """
    The following function prints the shape and size of the data frame
    by taking data parameter as pandas data frame
    frame_type as string [train_data, test_data, full_data or validation_data]
    """
    print(f" ----- Printing shape size and info of {frame_type} -----")
    print(f"The {frame_type} had a shape of", data.shape)
    print(f"The {frame_type} has a length of", len(data))
    print(data.info())


def get_number_of_classes_present(data: pd.DataFrame()):
    """
    Getting the number of classes present in the data set
    """
    group_by_label = data.groupby("labels").count()
    no_of_classes = list(group_by_label.index)
    print(f"Number of classes in the dataset {len(no_of_classes)}")


def display_random_image(image_df: pd.DataFrame(), n: int, random_seed: int, file: str):
    """
    The following function is taken from display_random_images
    function in the code
    https://www.kaggle.com/code/hoangtrung456/bird-classification
    """
    index_images = list(image_df.index)
    data_path = IMG_DATA
    # Checking if radom seed in present
    if random_seed:
        random.seed(random_seed)
    random_indexs = random.sample(index_images, n)
    plt.figure(figsize=(25, 15))
    plt.rcParams.update({"font.size": 7})
    for i, img_index in enumerate(random_indexs):
        ig = image_df.loc[img_index, "filepaths"]
        img_path = f"{data_path}/{ig}"
        label = image_df.loc[img_index, "labels"]
        # Show Image
        img = Image.open(img_path)
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(f"{label}")
        plt.axis("off")
    plt.savefig(f"{FIGS}/{file}.png")


def explore_data_image(data: pd.DataFrame(), label: str):
    """
    The following function is used for exploring the data frame
    """
    with open(f"{OUTPUTS}/image_data_analysis.txt", "a") as f:
        tem = sys.stdout
        sys.stdout = f
        print(f"---------- Exploring the {label} ----------")
        # Getting shape and size and info of the image data
        get_shape_size_and_info(data, label)
        # Checking if data has null values
        print("Different Object type present in dataset")
        print(data.dtypes)
        print("Checking if data has null values")
        print(data.isnull().sum())
        # Checking Number of classes present
        get_number_of_classes_present(data)
        label_count = data["labels"].value_counts().to_markdown()
        label_percentages = (data["labels"].value_counts(normalize=True) * 100).to_markdown()
        # Getting count of each data label in data
        print("Label count")
        print(label_count)
        # Getting the percentage of each label in data
        print("Label composition in dataset")
        print(label_percentages)
        # Display images from data
        sys.stdout = tem
        f.close()
    """
    The bellow function will generate random 5 images from the
    data provided
    """
    display_random_image(data, 5, 42, label)


def find_avg_sentence_length(sentences):
    """
    The following code is created by referring the
    "How long is each sentence on average" code in the
    following link
    https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb
    """
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    avg_length = np.mean(sentence_lengths)
    return avg_length


# The following function is written for Analysis of NLP data
def explore_text_data(data):
    """
    The following function explores the train, test and validation data
    of NLP data
    """
    train_data = data["train_data"]
    test_data = data["test_data"]
    valid_data = data["valid_data"]
    train_sentences = data["train_sentences"]
    test_sentences = data["test_sentences"]
    valid_sentences = data["valid_sentences"]
    train_vocabulary = data["train_vocabulary"]
    with open(f"{OUTPUTS}/text_data_analysis.txt", "a") as f:
        tem = sys.stdout
        sys.stdout = f
        # Finding the distribution of labels
        label_count_1 = train_data.target.value_counts()
        print("Count of labels present in train data")
        print(label_count_1)
        label_count_2 = test_data.target.value_counts()
        print("Count of labels present in test data")
        print(label_count_2)
        label_count_3 = valid_data.target.value_counts()
        print("Count of labels present in valid data")
        print(label_count_3)
        # Finding the distribution of number of lines in each abstract
        plt.hist(train_data.total_lines)
        plt.title("Distribution of abstract lengths")
        plt.xlabel("Abstract length")
        plt.ylabel("Frequency")
        plt.savefig(f"{IMG_DATA}/number_of_lines.png")
        plt.show()
        # Finding the counts of train, test and validation sentences
        print("Count of train sentences", len(train_sentences))
        print("Count of test sentences", len(test_sentences))
        print("Count of validation sentences", len(valid_sentences))
        # Finding the average sentence lengths for datasets
        print("Average sentence length of train data", find_avg_sentence_length(train_sentences))
        print("Average sentence length of test data", find_avg_sentence_length(test_sentences))
        print("Average sentence length of validation data", find_avg_sentence_length(valid_sentences))
        # Understanding the vocabulary of train data
        print("Number of words in vocabulary:", len(train_vocabulary))
        print("Top 10 common words", train_vocabulary[0:10])
        print("Least 10 common words", train_vocabulary[-10:-1])
        sys.stdout = tem
        f.close()
