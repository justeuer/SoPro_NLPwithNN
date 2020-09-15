import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from typing import Tuple


def create_model(input_dim,
                 embedding_dim,
                 context_dim,
                 output_dim):
    model = Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=embedding_dim))
    model.add(layers.SimpleRNN(context_dim))
    model.add(layers.Dense(output_dim))
    optimizer = tf.keras.optimizers.SGD()
    loss_object = tf.keras.losses.CosineSimilarity()
    model.compile(loss="cosine_similarity", optimizer=optimizer)
    return model, optimizer, loss_object


def create_deep_model(input_dim,
                      hidden_dim,
                      n_hidden,
                      output_dim):
    model = DeepModel(input_dim=input_dim, hidden_dim=hidden_dim, n_hidden=n_hidden, output_dim=output_dim)
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CosineSimilarity()
    model.compile(loss="cosine_similarity", optimizer=optimizer)
    model.build(input_shape=input_dim)
    return model, optimizer, loss_object


class DeepModel(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, n_hidden, output_dim):
        super(DeepModel, self).__init__()
        self.flatten = layers.Flatten(input_shape=input_dim)
        self.dense_layers = []
        for _ in range(n_hidden):
            self.dense_layers.append(layers.Dense(hidden_dim, activation='relu'))
        #self.recursive_layer = layers.SimpleRNN(units=128, input_shape=(1, 256))
        self.out = layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.flatten(inputs)
        for dl in self.dense_layers:
            x = dl(x)
        #x = self.recursive_layer(x)
        return self.out(x)


def cos_sim(x: np.array, y: np.array):
    return np.dot(x, y) / (norm(x) * norm(y))


