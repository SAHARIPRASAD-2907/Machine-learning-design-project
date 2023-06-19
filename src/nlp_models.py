import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from src.analysis_data import explore_text_data
from src.constants import NLP_DATA
from src.graphs import plot_model_history
from src.nlp_preprocessing import (
    get_vocab,
    label_encoding,
    pre_process_raw_text,
    tensor_texts,
)
from src.test_suit import my_test_suite
from tensorflow.keras import layers

# Setting File paths
TRAIN_FILE = NLP_DATA / "train.txt"
VALIDATE_FILE = NLP_DATA / "dev.txt"
TEST_FILE = NLP_DATA / "test.txt"
HUB_URL1 = "https://tfhub.dev/google/universal-sentence-encoder/4"
HUB_URL2 = "https://tfhub.dev/google/LEALLA/LEALLA-base/1"


def train_and_get_history(model, train_dataset, valid_dataset):
    """
    The following concept of steps_per_epochs was taken from
    https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb
    """
    history = model.fit(
        train_dataset,
        steps_per_epoch=int(0.1 * len(train_dataset)),
        epochs=10,
        validation_data=valid_dataset,
        validation_steps=int(0.1 * len(valid_dataset)),
    )
    return history


def print_test_suite(model, test_data, y_true, model_name):
    # Running custom test suite
    y_pred = np.argmax(model.predict(test_data), axis=1)
    my_test_suite(y_true, y_pred, model=model_name)


def train_and_get_scores(model, train_dataset, valid_dataset, test_dataset, test_la_encoded, model_name):
    history = train_and_get_history(model, train_dataset, valid_dataset)
    plot_model_history(history, "loss", "val_loss", model=f"{model_name}")
    plot_model_history(history, "accuracy", "val_accuracy", model=f"{model_name}")
    print_test_suite(model, test_dataset, test_la_encoded, model_name=f"{model_name}_tests")
    return model


def nlp_training_function():
    # Preprocess data and make data frame
    train_samples = pre_process_raw_text(TRAIN_FILE)
    validate_sample = pre_process_raw_text(VALIDATE_FILE)
    test_samples = pre_process_raw_text(TEST_FILE)
    # Creating Data frames of preprocessed samples
    train_data_frame = pd.DataFrame(train_samples)
    validation_data_frame = pd.DataFrame(validate_sample)
    test_data_frame = pd.DataFrame(test_samples)
    # Get sentences from data frame
    train_sentences = train_data_frame["text"].to_list()
    validation_sentences = validation_data_frame["text"].to_list()
    test_sentences = test_data_frame["text"].to_list()
    # Label encoding of target
    train_label_encoded = label_encoding(train_data_frame, "target")
    valid_label_encoded = label_encoding(validation_data_frame, "target")
    test_label_encoded = label_encoding(test_data_frame, "target")
    # Get text vectorizer from perprocessing
    text_vectorizer = tensor_texts(train_sentences)
    # Creating tensor dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_label_encoded))
    valid_dataset = tf.data.Dataset.from_tensor_slices((validation_sentences, valid_label_encoded))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_label_encoded))
    # Make tensors dataset into pre fetched batches
    train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    # Model 1 making custom token embedding layers
    text_vocab = get_vocab(text_vectorizer)
    token_embedding = layers.Embedding(
        input_dim=len(text_vocab), output_dim=128, mask_zero=True, name="token_embedding"
    )
    data = {
        "train_data": train_data_frame,
        "test_data": test_data_frame,
        "valid_data": validation_data_frame,
        "train_sentences": train_sentences,
        "test_sentences": test_sentences,
        "valid_sentences": validation_sentences,
        "train_vocabulary": text_vocab,
    }
    # Analyzing the data
    """
    Run this function only once and comment this function if
    you want to train the below models.
    """
    explore_text_data(data)
    one_dimensional_token_model(
        train_dataset,
        valid_dataset,
        test_dataset,
        test_label_encoded,
        text_vectorizer,
        token_embedding,
        model_name="TokenEmbededNLP",
    )
    # Model 2 Running the model with universal
    tf_hub_embedding_layer_1 = hub.KerasLayer(HUB_URL1, trainable=False, name="universal_sent_encoder")
    train_with_hub_layer(
        tf_hub_embedding_layer_1,
        train_dataset,
        valid_dataset,
        test_dataset,
        test_label_encoded,
        model_name="UniversalEmbededNLP",
    )
    # Model 3 Running with LEALLA layer
    tf_hub_embedding_layer_2 = hub.KerasLayer(HUB_URL2, trainable=False, name="lealla_sentence_encoder")
    # """
    # The Code should be run on Linux environment with CUDA support
    # (e.g. Colab, Kaagle,clusters, etc)
    # """
    train_with_hub_layer(
        tf_hub_embedding_layer_2,
        train_dataset,
        valid_dataset,
        test_dataset,
        test_label_encoded,
        model_name="LEALLAEmbededNLP",
    )


def one_dimensional_token_model(
    train_dataset, validation_dataset, test_dataset, test_label_encoded, text_vectorizer, token_embeddings, model_name
):
    """
    The following model is taken from the following link
    https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb
    """
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    text_vectors = text_vectorizer(inputs)
    token_embeddings = token_embeddings(text_vectors)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(token_embeddings)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    train_and_get_scores(
        model, train_dataset, validation_dataset, test_dataset, test_label_encoded, model_name=model_name
    )


def train_with_hub_layer(
    tf_hub_embedding_layers, train_dataset, valid_dataset, test_dataset, test_label_encoded, model_name
):
    """
    The following model is taken from the following link
    https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb
    """
    inputs = layers.Input(shape=[], dtype=tf.string)
    pretrained_embedding = tf_hub_embedding_layers(inputs)
    x = layers.Dense(128, activation="relu")(pretrained_embedding)
    outputs = layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    train_and_get_scores(model, train_dataset, valid_dataset, test_dataset, test_label_encoded, model_name=model_name)
