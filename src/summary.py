import json

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from src.constants import FIGS, OUTPUTS


def summarize_image_classification():
    """
    The following function summarizes all the results generated for the
    Image processing of sports data.
    """
    # Printing the scores achieved with accuracy
    model1 = json.load(open(OUTPUTS / "ResNet50__tests.json"))
    model2 = json.load(open(OUTPUTS / "EfficientNetV2__tests.json"))
    model3 = json.load(open(OUTPUTS / "InceptionV3__tests.json"))
    model4 = json.load(open(OUTPUTS / "VGG16__tests.json"))
    model5 = json.load(open(OUTPUTS / "MobileNetV2__tests.json"))
    results = pd.DataFrame(
        {
            "ResNet50": model1,
            "EfficientNetV2": model2,
            "InceptionV3": model3,
            "VGG16": model4,
            "MobileNetV2": model5,
        }
    )
    print(results.to_markdown())
    # Printing the plots generated for each model
    model1_loss = Image.open(FIGS / "ResNet50__loss.png")
    model1_acc = Image.open(FIGS / "ResNet50__accuracy.png")
    model2_loss = Image.open(FIGS / "EfficientNetV2__loss.png")
    model2_acc = Image.open(FIGS / "EfficientNetV2__accuracy.png")
    model3_loss = Image.open(FIGS / "InceptionV3__loss.png")
    model3_acc = Image.open(FIGS / "InceptionV3__accuracy.png")
    model4_loss = Image.open(FIGS / "VGG16__loss.png")
    model4_acc = Image.open(FIGS / "VGG16__accuracy.png")
    model5_loss = Image.open(FIGS / "MobileNetV2__loss.png")
    model5_acc = Image.open(FIGS / "MobileNetV2__accuracy.png")
    fig = plt.figure(figsize=(12, 8))
    rows = 5
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(model1_loss)
    plt.axis("off")
    plt.title("Loss over epochs for ResNet50 model")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(model1_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for ResNet50 model")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(model2_loss)
    plt.axis("off")
    plt.title("Loss over epochs for EfficientNetV2 model")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(model2_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for EfficientNetV2 model")

    fig.add_subplot(rows, columns, 5)
    plt.imshow(model3_loss)
    plt.axis("off")
    plt.title("Loss over epochs for InceptionV3 model")

    fig.add_subplot(rows, columns, 6)
    plt.imshow(model3_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for InceptionV3 model")

    fig.add_subplot(rows, columns, 7)
    plt.imshow(model4_loss)
    plt.axis("off")
    plt.title("Loss over epochs for VGG16 model")

    fig.add_subplot(rows, columns, 8)
    plt.imshow(model4_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for VGG16 model")

    fig.add_subplot(rows, columns, 9)
    plt.imshow(model5_loss)
    plt.axis("off")
    plt.title("Loss over epochs for MobileNetV2 model")

    fig.add_subplot(rows, columns, 10)
    plt.imshow(model5_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for MobileNetV2 model")
    plt.savefig(FIGS / "all_image_model_plots.png")
    plt.show()


def summarize_text_classification():
    """
    The following function summarizes all the results generated for the
    Image processing of sports data.
    """
    # Printing scores with accuracy
    print("Printing the results for NLP text classification")
    model1 = json.load(open(OUTPUTS / "TokenEmbededNLP_tests.json"))
    model2 = json.load(open(OUTPUTS / "UniversalEmbededNLP_tests.json"))
    model3 = json.load(open(OUTPUTS / "LEALLAEmbededNLP_tests.json"))
    results = pd.DataFrame(
        {
            "TokenEmbededNLP": model1,
            "UniversalEmbededNLP": model2,
            "LEALLAEmbededNLP": model3,
        }
    )
    print(results.to_markdown())
    # Printing the plots generated for each model
    model1_loss = Image.open(FIGS / "TokenEmbededNLP_loss.png")
    model1_acc = Image.open(FIGS / "TokenEmbededNLP_accuracy.png")
    model2_loss = Image.open(FIGS / "UniversalEmbededNLP_loss.png")
    model2_acc = Image.open(FIGS / "UniversalEmbededNLP_accuracy.png")
    model3_loss = Image.open(FIGS / "LEALLAEmbededNLP_loss.png")
    model3_acc = Image.open(FIGS / "LEALLAEmbededNLP_accuracy.png")

    fig = plt.figure(figsize=(12, 6))
    rows = 3
    columns = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(model1_loss)
    plt.axis("off")
    plt.title("Loss over epochs for TokenEmbeded model")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(model1_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for TokenEmbeded model")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(model2_loss)
    plt.axis("off")
    plt.title("Loss over epochs for UniversalEmbededNLP model")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(model2_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for UniversalEmbededNLP model")

    fig.add_subplot(rows, columns, 5)
    plt.imshow(model3_loss)
    plt.axis("off")
    plt.title("Loss over epochs for LEALLAEmbeded model")

    fig.add_subplot(rows, columns, 6)
    plt.imshow(model3_acc)
    plt.axis("off")
    plt.title("Accuracy over epochs for LEALLAEmbeded model")

    plt.savefig(FIGS / "all_NLP_model_plots.png")
    plt.show()
