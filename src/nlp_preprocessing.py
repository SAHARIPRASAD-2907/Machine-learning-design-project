"""
The following file contains data pre processing steps
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# taken from 3.2 in https://arxiv.org/pdf/1710.06071.pdf
MAX_TOKENS = 68000


def get_lines_from_text_file(file_name: str):
    """
    The below function is written by referring the following link
    “Read a File Line by Line in Python.” GeeksforGeeks, 21 Nov. 2019,
    https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/.
    The following function reads lines from text file and returns the line
    """
    with open(file_name, "r") as file:
        return file.readlines()


def pre_process_raw_text(file_name: str):
    """
    The following function is made by referring the following
    function "preprocess_text_with_line_numbers" github repository
    https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb

    The following function reads the lines in the text file
    and creates the following
    - Target (What the line means)
    - text (The line extracted)
    - line_number (The number of the line)
    """
    lines_in_text_file = get_lines_from_text_file(file_name)
    abstract_lines = ""
    abstract_of_samples = []

    for line in lines_in_text_file:
        if line.startswith("###"):
            abstract_lines = ""
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines()

            for abs_line_number, abs_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abs_line.split("\t")
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1]
                line_data["line_number"] = abs_line_number
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_of_samples.append(line_data)
        else:
            abstract_lines = abstract_lines + line
    return abstract_of_samples


def label_encoding(data: pd.DataFrame(), column: str):
    """
    Label encoding will be useful for prediction
    """
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(data[column].to_numpy())


def tensor_texts(train_sentences):
    # taken from 3.2 in https://arxiv.org/pdf/1710.06071.pdf
    """
    The following text vectorixation and training vocabulary was written by using
    https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb
    """
    MAX_TOKENS = 68000
    text_vectorization = TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=55)  # noqa
    text_vectorization.adapt(train_sentences)
    return text_vectorization


def get_vocab(text_vectorization):
    return text_vectorization.get_vocabulary()
