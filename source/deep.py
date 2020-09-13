from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from utils import create_deep_model
from classes import Alphabet, CognateSet, LevenshteinDistance

# set random seed for weights
tf.random.set_seed(seed=42)
encoding = None
HEADER_ROW = 0
COLUMN_SEPARATOR = ","
ID_COLUMN = 0
CONCEPT_COLUMN = 1
# online training, one cognate set a time
BATCH_SIZE = 1
MODELS = ['ipa', 'asjp', 'latin']


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data",
                        type=str,
                        default="../data/romance_asjp_full.csv",
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
    parser.add_argument("--n_hidden",
                        type=int,
                        default=1,
                        help="number of hidden layers")
    return parser.parse_args()


def main():
    global encoding
    args = parse_args()
    # determine whether to use the aligned or unaligned data
    assert args.aligned in [0, 1], "Too many instances of --aligned switch, should be 0 or 1"
    aligned = bool(args.aligned)
    # load data
    data_file = Path(args.data)
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
    alphabet = Alphabet(alphabet_file, encoding=encoding, orthographic=False)
    assert isinstance(args.epochs, int), "Epochs not int, but {}".format(type(args.epochs))
    assert args.epochs > 0, "Epochs out of range: {}".format(args.epochs)
    epochs = args.epochs
    # number of hidden layers
    assert args.n_hidden > 0, "Number of hidden layers should be at least 1 ;)"
    n_hidden = args.n_hidden
    # determine ancestor
    ancestor = args.ancestor
    print("alphabet:")
    print(alphabet)


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
        cognate_set = CognateSet(id=id,
                        concept=concept,
                        ancestor=ancestor,
                        cognate_dict=cognate_dict,
                        alphabet=alphabet)
        cognate_sets.append(cognate_set)

    #cognate_sets = cognate_sets[:10]
    batches = len(cognate_sets)

    # input shape is a vector of size (number of languages without the ancestor * number of features)
    input_dim = (1, len(langs) - 1, alphabet.get_feature_dim())
    # output dim is the number of characters (for classification)
    output_dim = alphabet.get_feature_dim()
    #context dim is the 
    context_dim = 128
    # define model
    model, optimizer, loss_object = create_deep_model(input_dim=input_dim,
                                                      hidden_dim=256,
                                                      n_hidden=n_hidden,
                                                      units=context_dim,
                                                      output_dim=output_dim)

    model.summary()


    words_true = []
    words_pred = []
    batch_losses = []

    # train
    for epoch in range(1, epochs+1):
        words_true.clear()
        words_pred.clear()
        for batch, cognate_set in enumerate(cognate_sets):
            output_characters = []
            for lang_array in cognate_set:
                target = tf.keras.backend.expand_dims(lang_array.pop(ancestor).to_numpy(), axis=0)
                target = tf.dtypes.cast(target, tf.float32)
                data = []
                for lang, vec in lang_array.items():
                    data.append(list(vec))
                data = np.array(data)
                data = tf.dtypes.cast(data, tf.float32)
                data = tf.reshape(data, (1, -1))
                with tf.GradientTape(persistent=True) as tape:
                    output = model(data)
                    loss = loss_object(target, output)
                    batch_losses.append(float(loss))
                    gradients = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                    output_characters.append(alphabet.get_char_by_feature_vector(output))
            words_pred.append("".join(output_characters))
            words_true.append(str(cognate_set.get_ancestor()))
            mean_loss = np.mean(batch_losses)
            if batch % 10 == 0:
                print("Epoch[{}]/[{}], Batch[{}]/[{}], mean loss = {}".format(epoch, epochs, batch, batches, mean_loss))

        # calculate levenshtein distance
        ld = LevenshteinDistance(true=words_true, pred=words_pred)
        ld.print_distances()
        ld.print_percentiles()


if __name__ == '__main__':
    main()
