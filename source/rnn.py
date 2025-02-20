import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import sys

from classes import Alphabet, CognateSet, LevenshteinDistance
from utils import create_model

# create output dir
out_dir = Path("../data/out")
if not out_dir.exists():
    out_dir.mkdir()

# set random seed for weights
tf.random.set_seed(seed=42)
encoding = None
HEADER_ROW = 0
COLUMN_SEPARATOR = ","
ID_COLUMN = 0
CONCEPT_COLUMN = 1
# online training, one cognate set a time
BATCH_SIZE = 1
MODELS = ['asjp', 'ipa', 'latin']
valid_size = 0.8
batch_size = 1

output_characters = []


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data",
                        type=str,
                        default="../data/romance_swadesh_asjp.csv",
                        help="file containing the cognate sets")
    parser.add_argument("--model",
                        type=str,
                        default="asjp",
                        help="model to be trained")
    parser.add_argument("--ancestor",
                        type=str,
                        default='latin',
                        help="column corresponding to the proto-form")
    parser.add_argument("--orthographic",
                        default=0,
                        action='count',
                        help="switch between orthographic and feature-based model")
    parser.add_argument("--aligned",
                        default=0,
                        action='count',
                        help="switch between aligned and unaligned model")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="number of epochs")
    return parser.parse_args()


def main():
    global encoding
    args = parser_args()
    # determine whether the model should use feature encodings or character embeddings
    assert args.orthographic in [0, 1], "Too many instances of --orthographic switch, should be 0 or 1"
    orthographic = bool(args.orthographic)
    # determine whether to use the aligned or unaligned data
    assert args.aligned in [0, 1], "Too many instances of --aligned switch, should be 0 or 1"
    aligned = bool(args.aligned)
    # load data
    data_file = None
    if args.data == "ipa":
    	encoding = 'utf-16'
    	data_file = Path("../data/romance_swadesh_ipa.csv")
    elif args.data == "asjp":
    	encoding = 'ascii'
    	data_file = Path("../data/romance_swadesh_asjp.csv")
    assert data_file.exists() and data_file.is_file(), "Data file {} does not exist".format(data_file)
    # determine model
    assert args.model in MODELS, "Model should be one of {}".format(MODELS)
    # determine path to alphabet file & encoding
    alphabet_file = None
    if args.model == "ipa":
        encoding = 'utf-16'
        alphabet_file = Path("../data/alphabets/ipa.csv")
    elif args.model == "asjp":
        encoding = 'ascii'
        alphabet_file = Path("../data/alphabets/asjp.csv")
    # load data from file
    assert alphabet_file.exists() and alphabet_file.is_file(), "Alphabet file {} does not exist".format(alphabet_file)
    alphabet = Alphabet(alphabet_file, encoding=encoding, orthographic=orthographic)
    assert isinstance(args.epochs, int), "Epochs not int, but {}".format(type(args.epochs))
    assert args.epochs > 0, "Epochs out of range: {}".format(args.epochs)
    epochs = args.epochs
    print("alphabet:")
    print(alphabet)

    # initialize model
    model, optimizer, loss_object = create_model(input_dim=alphabet.get_feature_dim(),
                                                 embedding_dim=28,
                                                 context_dim=128,
                                                 output_dim=alphabet.get_feature_dim())

    model.summary()

    print("data_file: {}".format(data_file.absolute()))
    print("model: {}, orthographic={}, aligned={}".format(args.model, orthographic, aligned))
    print("alphabet: {}, read from {}".format(args.model, alphabet_file.absolute()))
    print("epochs: {}".format(epochs))

    # create cognate sets

    cognate_sets = []

    data = data_file.open(encoding='utf-16').read().split("\n")
    cols = data[HEADER_ROW].split(COLUMN_SEPARATOR)
    langs = cols[2:]

    for li, line in enumerate(data[HEADER_ROW:]):
        if aligned:
            if line == "" or li % 2 != 0:
                continue
        else:
            if line == "" or li % 2 == 0:
                continue
        row_split = line.split(COLUMN_SEPARATOR)
        id = row_split[ID_COLUMN]
        concept = row_split[CONCEPT_COLUMN]
        words = row_split[CONCEPT_COLUMN + 1:]
        cognate_dict = {}
        assert len(langs) == len(words), "Langs / Words mismatch, expected {}, got {}".format(len(langs), len(words))
        for lang, word in zip(langs, words):
            cognate_dict[lang] = alphabet.translate(word)
        cs = CognateSet(id=id,
                        concept=concept,
                        ancestor='latin',
                        cognate_dict=cognate_dict,
                        alphabet=alphabet)
        cognate_sets.append(cs)

    # maybe we needn't do the evaluation, since we mainly want to know how
    # the model behaves with the different inputs

    #split_index = int(valid_size * len(cognate_sets))
    #train_data = cognate_sets[:split_index]
    #valid_data = cognate_sets[split_index:]
    #print("train size: {}".format(len(train_data)))
    #print("valid size: {}".format(len(valid_data)))
    #cognate_sets = cognate_sets[10:30]

    words_true = []
    words_pred = []
    epoch_losses = []
    batch_losses = []

    for epoch in range(epochs):
        # reset lists
        epoch_losses.clear()
        words_true.clear()
        words_pred.clear()
        # iterate over the cognate sets
        for i, cs in enumerate(cognate_sets):
            # reset batch loss
            batch_losses.clear()
            # iterate over the character embeddings
            for j, char_embeddings in enumerate(cs):
                # add a dimension to the latin character embedding (ancestor embedding)
                # we add a dimension because we use a batch size of 1 and TensorFlow does not
                # automatically insert the batch size dimension
                target = tf.keras.backend.expand_dims(char_embeddings.pop(cs.ancestor).to_numpy(), axis=0)
                # convert the latin character embedding to float32 to match the dtype of the output (line 137)
                target = tf.dtypes.cast(target, tf.float32)
                # iterate through the embeddings
                # initialize the GradientTape
                with tf.GradientTape(persistent=True) as tape:
                    for lang, embedding in char_embeddings.items():
                        # add a dimension to the the embeddings
                        data = tf.keras.backend.expand_dims(embedding.to_numpy(), axis=0)
                        output = model(data)
                        # calculate the loss
                        loss = loss_object(target, output)
                        epoch_losses.append(float(loss))
                        batch_losses.append(float(loss))
                        # calculate the gradients
                        gradients = tape.gradient(loss, model.trainable_weights)
                        # backpropagate
                        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                        # convert the character vector into a character
                    output_char = alphabet.get_char_by_feature_vector(output)
                    # append the converted vectors to a list so we can see the reconstructed word
                    output_characters.append(output_char)
            # append the reconstructed word and the ancestor to the true/pred lists
            words_pred.append("".join(output_characters))
            words_true.append(str(cs.get_ancestor()))
            # clear the list of output characters so we can create another word
            output_characters.clear()
            print("Batch {}, mean loss={}".format(i, np.mean(batch_losses)))
        # calculate distances
        ld = LevenshteinDistance(true=words_true,
                                 pred=words_pred)
        print("Epoch {} finished".format(epoch + 1))
        print("Mean loss={}".format(epoch, np.mean(epoch_losses)))
        ld.print_distances()
        ld.print_percentiles()

    # do so again after training has finished, but now also save the plots
    ld = LevenshteinDistance(true=words_true,
                             pred=words_pred)
    ld.print_distances()
    ld.print_percentiles()
    ld.plot_distances(Path("../data/out/distances.png"))
    ld.plot_percentiles(Path("../data/out/percentiles.png"))


if __name__ == '__main__':
    main()
