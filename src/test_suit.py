import json

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.constants import OUTPUTS


def my_test_suite(y_true, y_pred, model):
    """
    The following my_test_suite takes the following parameters as input and is made by referring sklearn documentation
    y_true: numpy array
    y_pred: numpy arrray
    model: string
    It creates a json file with the values and returns the values of score
    """
    # Calculating accuracy
    accuracy_of_model = accuracy_score(y_true, y_pred)
    # Calculate precision
    precision_of_model = precision_score(y_true, y_pred, average="weighted")
    # Calculate recall
    recall_of_model = recall_score(y_true, y_pred, average="weighted")
    # Calculate F1 score
    f1_score_of_model = f1_score(y_true, y_pred, average="weighted")
    scores = {
        "model_accuracy": accuracy_of_model,
        "model_precision": precision_of_model,
        "model_recall": recall_of_model,
        "model_f1": f1_score_of_model,
    }
    with open(f"{OUTPUTS}/{model}.json", "w") as outfile:
        json.dump(scores, outfile)
    return scores
