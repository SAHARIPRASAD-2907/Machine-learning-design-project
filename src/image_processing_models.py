import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.constants import IMG_DATA
from src.graphs import plot_model_history
from src.test_suit import my_test_suite
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setting All global paths
IMAGE_SIZE = 150
BATCH_SIZE = 32


# Setting data paths
data_file = IMG_DATA / "sports.csv"
train_dir = IMG_DATA / "train"
test_dir = IMG_DATA / "test"
val_dir = IMG_DATA / "valid"

train_data_gen = ImageDataGenerator()
val_data_gen = ImageDataGenerator()
test_data_gen = ImageDataGenerator()

train_img = train_data_gen.flow_from_directory(
    train_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True, class_mode="categorical"
)
validation_img = val_data_gen.flow_from_directory(
    val_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False, class_mode="categorical"
)
test_imag = test_data_gen.flow_from_directory(
    test_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=False, class_mode="categorical"
)


def train_and_get_history(model):
    history = model.fit(train_img, epochs=10, validation_data=validation_img)
    return history


def print_test_suite(model, model_name):
    """
    The following function takes the trained model
    """
    # Running custom test suite
    data = pd.read_csv(data_file)
    test_data = data[data["data set"] == "test"].reset_index()
    y_pred = np.argmax(model.predict(test_imag), axis=1)
    y_true = LabelEncoder().fit_transform(test_data["labels"])
    my_test_suite(y_true, y_pred, model=model_name)


def train_and_get_scores(model, model_name):
    history = train_and_get_history(model)
    plot_model_history(history, "loss", "val_loss", model=f"{model_name}")
    plot_model_history(history, "accuracy", "val_accuracy", model=f"{model_name}")
    print_test_suite(model, model_name=f"{model_name}_tests")
    return model


def configure_model(model):
    """
    The following parameters was taken from the following link for as reference
    https://www.kaggle.com/code/shamsheerrahiman/cnn-birds-resnet50
    """
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_resnet50():
    model = Sequential()
    model.add(
        ResNet50(
            include_top=False, pooling="avg", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", classes=100
        )
    )
    # Configuring model
    model = configure_model(model)
    train_and_get_scores(model, model_name="ResNet50_")


def train_mobile_net_v2():
    model = Sequential(
        MobileNetV2(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", classes=100)
    )
    model = configure_model(model)
    train_and_get_scores(model, model_name="MobileNetV2_")


def train_VGG16():
    model = Sequential(
        VGG16(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", classes=100)
    )
    model = configure_model(model)
    train_and_get_scores(model, model_name="VGG16_")


def train_EfficientNetV2L():
    model = Sequential(
        EfficientNetV2L(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", classes=100)
    )
    model = configure_model(model)
    train_and_get_scores(model, model_name="EfficientNetV2L_")


def train_InceptionV3():
    model = Sequential(
        InceptionV3(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights="imagenet", classes=100)
    )
    model = configure_model(model)
    train_and_get_scores(model, model_name="InceptionV3_")
