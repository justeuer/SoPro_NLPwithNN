from copy import deepcopy
import numpy as np
from numpy.linalg import norm
from pathlib import Path
import random
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from typing import List, Set


def create_model(input_dim,
                 embedding_dim,
                 context_dim,
                 output_dim):
    model = Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=embedding_dim))
    model.add(layers.SimpleRNN(context_dim))
    model.add(layers.Dense(output_dim, activation='sigmoid'))
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
    # cosine similarity since that is also the function we use to retrieve the chars
    # from the model output.
    loss_object = tf.keras.losses.CosineSimilarity()
    model.compile(loss="cosine_similarity", optimizer=optimizer)
    model.build(input_shape=input_dim)
    return model, optimizer, loss_object


def create_lstm_model(input_dim,
                      embedding_dim,
                      context_dim,
                      output_dim):
    model = Sequential()
    model.add(layers.Embedding(input_dim=input_dim, output_dim=embedding_dim))
    model.add(layers.SimpleRNN(context_dim))
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CosineSimilarity()
    model.compile(loss="cosine_similarity", optimizer=optimizer)
    return model, optimizer, loss_object


def create_many_to_one_model(lstm_dim, timesteps, data_dim, fc_dim, output_dim):
    """
    Creates a many - to - one model with LSTM units.
    Parameters
    ----------
    lstm_dim
        Size of the LSTM layer
    timesteps
        Number of timesteps = number of inputs into the LSTM layer = number of romance language chars
    data_dim
        Size of the character/feature embedding
    fc_dim
        Size of the intermediate fully connected layer
    output_dim
        Size of the output layer = size of the charcter/feature embedding
    Returns
        The compiled model, the loss object, and the optimizer
    -------

    """
    model = Sequential()
    # This creates an LSTM layer that expects 5 inputs (which is, five characters in the five
    # romance languages). We could use 5 separate layers though
    model.add(layers.LSTM(lstm_dim, activation='relu', input_shape=(timesteps, data_dim)))
    # First dense layer, to avoid narrowing of the signal from 128 to 10 in one step
    model.add(layers.Dense(fc_dim, activation='relu'))
    # output layer, size of the feature/character embedding
    model.add(layers.Dense(output_dim, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CosineSimilarity()
    model.compile(loss="cosine_similarity", optimizer=optimizer)
    return model, optimizer, loss_object


class DeepModel(tf.keras.Model):
    """
    Deep feedforward network. We agglutinate all character vectors at one position
    into a single vector and then feed that into 2 fully connected layers (so technically
    there is only one real 'deep' layer.
    """

    def __init__(self, input_dim, hidden_dim, n_hidden, output_dim):
        super(DeepModel, self).__init__()
        # project into a single vector
        self.flatten = layers.Flatten(input_shape=input_dim)
        # create as many hidden layers as specified & append them to an internal list
        self.dense_layers = []
        for _ in range(n_hidden):
            self.dense_layers.append(layers.Dense(hidden_dim, activation='relu'))
        # we chose sigmoid as the activation function since the desired output is a multi-
        # hot vector (or a one-hot vector in the case of char embeddings)
        self.out = layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs, **kwargs):
        # projection
        x = self.flatten(inputs)
        # apply fully connected layers
        for dl in self.dense_layers:
            x = dl(x)
        return self.out(x)


def cos_sim(x: np.array, y: np.array):
    return np.dot(x, y) / (norm(x) * norm(y))


def list_to_str(lst: List[float]):
    s = ""
    for num in lst:
        s += str(int(num)) + ","
    return s[:len(s) - 1]


def train_test_split_ids(n: int, tag: str, valid_size: 0.2):
    """
    Creates files containing the train/test indices for a given corpis
    Parameters
    ----------
    n
        The size of the corpus
    tag
        The tag (name) of the corpus
    valid_size
        The size of the test set
    Returns
    -------
    """
    # tag cannot be null
    assert tag is not None and tag != "", "Tag cannot be None!"

    # determine indices
    train_size = int((1 - valid_size) * n)
    indices = set(range(1, n + 1))
    train_indices = set(random.sample(indices, train_size))
    test_indices = indices.difference(train_indices)

    # paths
    outpath_train = Path("../data/{}_train_indices.txt".format(tag))
    outpath_test = Path("../data/{}_test_indices.txt".format(tag))
    # create files
    outpath_train.touch()
    outpath_test.touch()

    # save indices for later use
    # train
    outfile_train = outpath_train.open(mode='w')
    for index in train_indices:
        outfile_train.write("{}\n".format(index))
    outfile_train.close()
    print("Train indices saved at {}".format(outpath_train.absolute()))

    # test
    outfile_test = outpath_test.open(mode='w')
    for index in test_indices:
        outfile_test.write("{}\n".format(index))
    outfile_test.close()
    print("Test indices saved at {}".format(outpath_test.absolute()))


def cross_validation_runs(n: int, indices: Set[str]):
    """
    Will randomly sample n cross-validation folds from a set of indices
    Parameters
    ----------
    n
        The number of cross-validation folds
    indices
        The indices to be sampled from
    Returns
        A list of combinations of the n folds, for n cross-validation runs
    -------

    """
    folds = []
    # This means that indices beyond n * fold_size will be ignored
    fold_size = int(len(indices) / n)
    print("Preparing {} folds of size {}".format(n, fold_size))

    for _ in range(n):
        fold = random.sample(indices, fold_size)
        indices = indices.difference(fold)
        folds.append(fold)

    # now sample n runs from the n folds
    runs = []
    for i in range(n):
        folds_ = deepcopy(folds)
        test_data = folds_.pop(i)
        train_data = []
        for fold in folds_:
            train_data.extend(fold)
        runs.append({
            'train': train_data,
            'test': test_data
        })

    return runs

# if __name__ == '__main__':
# train_test_split_ids(100, "swadesh", valid_size=0.0)
# train_test_split_ids(3218, "ciobanu", valid_size=0.2)
# indices = set([str(i) for i in range(1, 3219)])
# runs = cross_validation_runs(5, indices)
# print(runs)
