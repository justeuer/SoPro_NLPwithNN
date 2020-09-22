from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import random

from utils import create_many_to_one_model
from classes import Alphabet, CognateSet, LevenshteinDistance
from plots import plot_results

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
                        default="../data/romance_ipa_full.csv",
                        help="file containing the cognate sets")
    parser.add_argument("--model",
                        type=str,
                        default="asjp",
                        help="model to be trained")
    parser.add_argument("--ancestor",
                        type=str,
                        default='latin',
                        help="column corresponding to the proto-form")
    parser.add_argument("--ortho",
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
    parser.add_argument("--out_tag",
                        type=str,
                        default="swadesh",
                        help="tag for output directories")
    return parser.parse_args()


def main():
    global encoding

    args = parse_args()

    # determine whether to use the aligned or unaligned data
    assert args.aligned in [0, 1], "Too many instances of --aligned switch, should be 0 or 1"
    aligned = bool(args.aligned)

    # and decide between feature encodings and character embeddings
    assert args.ortho in [0, 1], "Too many instances of --ortho switch, should be 0 or 1"
    ortho = bool(args.ortho)

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
    elif args.model == 'latin':
        encoding = 'utf-16'
        alphabet_file = Path("../data/alphabets/latin.csv")
    # load data from file
    assert alphabet_file.exists() and alphabet_file.is_file(), "Alphabet file {} does not exist".format(alphabet_file)
    alphabet = Alphabet(alphabet_file, encoding=encoding, ortho=ortho)

    # number of epochs
    assert isinstance(args.epochs, int), "Epochs not int, but {}".format(type(args.epochs))
    assert args.epochs > 0, "Epochs out of range: {}".format(args.epochs)
    epochs = args.epochs

    # number of hidden layers
    # assert args.n_hidden > 0, "Number of hidden layers should be at least 1 ;)"
    # n_hidden = args.n_hidden

    # determine output directories, create them if they do not exist
    out_tag = "_{}".format(args.out_tag)
    plots_dir = Path("../out/plots{}_many2one".format(out_tag))
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True)
    results_dir = Path("../out/results{}_many2one".format(out_tag))
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    # create file for results
    result_file_path = results_dir / "m2one_{}{}{}.txt".format(args.model,
                                                               "_aligned" if aligned else "",
                                                               "_ortho" if ortho else "")
    result_file_path.touch()
    result_file = result_file_path.open('w', encoding=encoding)

    # determine ancestor
    ancestor = args.ancestor

    # create cognate sets
    cognate_sets = []
    data = data_file.open(encoding='utf-16').read().split("\n")
    cols = data[HEADER_ROW].split(COLUMN_SEPARATOR)
    langs = cols[2:]

    # import tensorflow here to comply with the wiki entry https://wiki.lsv.uni-saarland.de/doku.php?id=cluster
    import tensorflow as tf
    # set random seed for weights
    tf.random.set_seed(seed=42)

    for li, line in enumerate(data[HEADER_ROW:]):
        if args.model == 'latin':
            if line == "":
                continue
        elif aligned:
            if line == "" or li % 2 == 0:
                continue
        else:
            if line == "" or li % 2 != 0:
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

    # prepare train_test_split
    data = {i: cognate_set for i, cognate_set in enumerate(cognate_sets)}
    train_size = 0.8
    n_train_samples = int(train_size * len(cognate_sets))
    train_indices = random.sample(list(data), n_train_samples)
    train_data = {i: cognate_set for i, cognate_set in data.items() if i in train_indices}
    test_data = {i: cognate_set for i, cognate_set in data.items() if i not in train_indices}

    print("Train size: {}".format(len(train_data)))
    print("Test size: {}".format(len(test_data)))

    # define model
    model, optimizer, loss_object = create_many_to_one_model(lstm_dim=128,
                                                             timesteps=len(langs) - 1,
                                                             data_dim=alphabet.feature_dim,
                                                             fc_dim=100,
                                                             output_dim=alphabet.feature_dim)
    model.summary()

    words_true = []
    words_pred = []
    epoch_losses = []
    batch_losses = []

    # Training
    print("***** Start training *****")
    for epoch in range(1, epochs + 1):
        words_true.clear()
        words_pred.clear()
        batch_losses.clear()
        for batch, cognate_set in train_data.items():
            output_characters = []
            for lang_array in cognate_set:
                target = tf.keras.backend.expand_dims(lang_array.pop(ancestor).to_numpy(), axis=0)
                target = tf.dtypes.cast(target, tf.float32)
                data = []
                for lang, vec in lang_array.items():
                    data.append(list(vec))
                data = np.array(data)
                data = tf.keras.backend.expand_dims(data, axis=0)
                data = tf.dtypes.cast(data, tf.float32)
                # data = tf.reshape(data, (1, -1))
                with tf.GradientTape() as tape:
                    output = model(data)
                    loss = loss_object(target, output)
                    batch_losses.append(float(loss))
                    gradients = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                    output_characters.append(alphabet.get_char_by_vector(output))
            words_pred.append("".join(output_characters))
            words_true.append(str(cognate_set.ancestor_word))
            # print("".join(output_characters), str(cognate_set.ancestor_word))
            if batch % 100 == 0:
                print("Epoch [{}/{}], Batch [{}/{}]".format(epoch, epochs, batch, len(cognate_sets)))
        # calculate mean epoch loss
        mean_loss = np.mean(batch_losses)
        epoch_losses.append(mean_loss)
        print("Epoch[{}]/[{}], mean batch loss = {}".format(epoch, epochs, mean_loss))
        # calculate levenshtein distance
        ld = LevenshteinDistance(true=words_true, pred=words_pred)
        ld.print_distances()
        ld.print_percentiles()

        # plot if it's the last epoch
        if epoch == epochs:
            outfile = plots_dir / "many2one_train_{}{}{}.jpg".format(args.model, "_aligned" if aligned else "",
                                                                     "_ortho" if ortho else "")
            title = "Model [Train]: many to one with LSTM, {}{}{}".format(", " + args.model,
                                                                          ", aligned" if aligned else "",
                                                                          ", orthographic" if ortho else "")
            plot_results(title=title,
                         distances={"=<" + str(d): count for d, count in ld.distances.items()},
                         percentiles={"=<" + str(d): perc for d, perc in ld.percentiles.items()},
                         mean_dist=ld.mean_distance,
                         mean_dist_norm=ld.mean_distance_normalized,
                         losses=epoch_losses,
                         outfile=Path(outfile))
            # save reconstructed words (but only if the edit distance is at least one)
            import nltk
            for t, p in zip(words_true, words_pred):
                distance = nltk.edit_distance(t, p)
                if distance > 0:
                    line = "{},{},distance={}\n".format(t, p, nltk.edit_distance(t, p))
                    result_file.write(line)
            result_file.close()
    words_pred.clear()
    words_true.clear()
    print("***** Training finished *****")
    print()

    # Testing
    # test_loss = []
    print("***** Start testing *****")
    for i, cognate_set in test_data.items():
        output_characters = []
        for lang_array in cognate_set:
            target = tf.keras.backend.expand_dims(lang_array.pop(ancestor).to_numpy(), axis=0)
            target = tf.dtypes.cast(target, tf.float32)
            data = []
            for lang, vec in lang_array.items():
                data.append(list(vec))
            data = np.array(data)
            data = tf.keras.backend.expand_dims(data, axis=0)
            data = tf.dtypes.cast(data, tf.float32)
            output = model(data)
            # loss = loss_object(target, output)
            output_characters.append(alphabet.get_char_by_vector(output))
        words_pred.append("".join(output_characters))
        words_true.append(str(cognate_set.ancestor_word))

    # create plots
    ld = LevenshteinDistance(words_true, words_pred)
    ld.print_distances()
    ld.print_percentiles()
    outfile = plots_dir / "many2one_test_{}{}{}.jpg".format(args.model, "_aligned" if aligned else "",
                                                            "_ortho" if ortho else "")
    title = "Model [Test]: many to one with LSTM, {}{}{}".format(", " + args.model, ", aligned" if aligned else "",
                                                                 ", orthographic" if ortho else "")
    plot_results(title=title,
                 distances={"=<" + str(d): count for d, count in ld.distances.items()},
                 percentiles={"=<" + str(d): perc for d, perc in ld.percentiles.items()},
                 mean_dist=ld.mean_distance,
                 mean_dist_norm=ld.mean_distance_normalized,
                 losses=[],
                 outfile=Path(outfile),
                 testing=True)

    print("***** Testing finished *****")


if __name__ == '__main__':
    main()
