import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras import layers, Sequential


def create_model(input_dim, embedding_dim, context_dim, output_dim):
    model = Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=embedding_dim))
    model.add(layers.SimpleRNN(context_dim))
    model.add(layers.Dense(output_dim))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_object = tf.keras.losses.CosineSimilarity()
    model.compile(loss="cosine_similarity", optimizer=optimizer)
    return model, optimizer, loss_object


def cos_sim(x: np.array, y: np.array):
    return np.dot(x, y) / (norm(x) * norm(y))


def build_model(input_dim, embedding_dim, lstm_dim, output_dim, langs):
    input_layers = []
    lstm_layers = []
    for lang in langs:
        input_layer = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, name=lang)
        lstm_layers.append(tf.keras.layers.LSTM(lstm_dim, return_sequences=False)(input_layer))
        input_layers.append(input_layer)

    output = tf.keras.layers.concatenate(inputs=lstm_layers, axis=1)
    output = tf.keras.layers.Dense(output_dim, activation='relu', name='output')(output)
    model = tf.keras.models.Model(inputs=input_layers, output=[output])

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='cosine-similarity', optimizer=optimizer)
    return model, optimizer
