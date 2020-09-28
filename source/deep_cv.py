from argparse import ArgumentParser
import numpy as np
from pathlib import Path

from utils import create_deep_model, cross_validation_runs
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


# To train the script with latin characters I run this command:
# python deep.py --data=../data/romance_orthographic.csv --model=latin --ortho

# and for the ciobanu data & latin alphabet:
# python deep.py --data=../data/romance_ciobanu_latin_orthographic.csv --model=latin --ancestor=ancestor \
# --out_tag=ciobanu
# The last switch will create separate output folders for the ciobanu data, but you can use any value
# If you don't want to overwrite existing files

# --ortho tells the script to use character embeddings. If it's not present, feature encodings will be used.

# --aligned will choose every second line instead, which contain the aligned cognate set.
# You can actually use --aligned and --ortho together, will result in the best model for each
# configuration.


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
    parser.add_argument("--n_hidden",
                        type=int,
                        default=2,
                        help="number of hidden layers")
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
    assert args.n_hidden > 0, "Number of hidden layers should be at least 1 ;)"
    n_hidden = args.n_hidden

    # determine output directories, create them if they do not exist
    out_tag = "_{}".format(args.out_tag)
    # and tag for files with train/test indices
    indices_tag = args.out_tag
    plots_dir = Path("../out/plots{}_deep".format(out_tag))
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True)
    # directory for lists of reconstructed vs. latin words
    results_dir = Path("../out/results{}_deep".format(out_tag))
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    # create file for results
    result_file_path = results_dir / "deep_{}{}{}.txt".format(args.model,
                                                              "_aligned" if aligned else "",
                                                              "_ortho" if ortho else "")
    #result_file_path.touch()
    #result_file = result_file_path.open('w', encoding=encoding)

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

    # start data extraction
    for li, line in enumerate(data[HEADER_ROW:]):
        # have to do that because the file with the latin characters doesn't contain aligned cognate sets
        if args.model == 'latin':
            if line == "":
                continue
        # but the other two do
        elif aligned:
            if line == "" or li % 2 == 0:
                print(line)
                continue
        # the unaligned case
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

    # prepare cross validation runs
    total_data = {str(i+1): cognate_set for i, cognate_set in enumerate(cognate_sets)}
    train_indices = set(total_data.keys())
    runs = cross_validation_runs(5, train_indices)
    #test_indices = Path("../data/{}_test_indices.txt".format(indices_tag)).open('r').read().split("\n")
    #train_data = {i: cognate_set for i, cognate_set in data.items() if i in train_indices}
    #test_data = {i: cognate_set for i, cognate_set in data.items() if i in test_indices}

    # input shape is a vector of size (number of languages without the ancestor * number of features)
    print("langs", langs)
    input_dim = (1, len(langs) - 1, alphabet.feature_dim)
    print(input_dim)
    # output dim is the number of characters (for classification)
    output_dim = alphabet.feature_dim
    # define model
    model, optimizer, loss_object = create_deep_model(input_dim=input_dim,
                                                      hidden_dim=256,
                                                      n_hidden=n_hidden,
                                                      output_dim=output_dim)
    model.summary()

    # save model weights for reset
    initital_weights = model.get_weights()

    words_true = []
    wts = []
    wps = []
    words_pred = []
    epoch_losses = []
    batch_losses = []

    # Training with cross-validation
    for i, run in enumerate(runs):
        print("***** Cross-validation run [{}/{}] *****".format(i+1, len(runs)))
        # reload initial model weights
        model.set_weights(initital_weights)
        # get train & test folds
        train_data = {i: cognate_set for i, cognate_set in total_data.items() if i in run['train']}
        test_data = {i: cognate_set for i, cognate_set in total_data.items() if i in run['test']}

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
                    data = tf.dtypes.cast(data, tf.float32)
                    data = tf.reshape(data, (1, -1))
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
                if int(batch) % 100 == 0:
                    print("Epoch [{}/{}], Batch [{}/{}]".format(epoch, epochs, batch, len(cognate_sets)))
            # calculate mean epoch loss
            mean_loss = np.mean(batch_losses)
            epoch_losses.append(mean_loss)
            print("Epoch[{}]/[{}], mean batch loss = {}".format(epoch, epochs, mean_loss))
            # calculate levenshtein distance
            ld = LevenshteinDistance(true=words_true, pred=words_pred)
            ld.print_distances()
            ld.print_percentiles()

            # save reconstructed words (but only if the edit distance is at least one)
            #import nltk
            #for t, p in zip(words_true, words_pred):
            #    distance = nltk.edit_distance(t, p)
            #    if distance > 0:
            #        line = "{},{},distance={}\n".format(t, p, nltk.edit_distance(t, p))
            #        result_file.write(line)
            #result_file.close()

        words_pred.clear()
        words_true.clear()
        print("***** Training finished *****")
        print()

        # Testing
        # Do the same thing as above with the test data, but don't collect the gradients
        # and don't backpropagate
        print("***** Start testing *****")
        for i, cognate_set in test_data.items():
            output_characters = []
            for lang_array in cognate_set:
                # drop the latin column to get right input size
                lang_array.pop(ancestor)
                # target = tf.keras.backend.expand_dims(lang_array.pop(ancestor).to_numpy(), axis=0)
                # target = tf.dtypes.cast(target, tf.float32)
                data = []
                for lang, vec in lang_array.items():
                    data.append(list(vec))
                data = np.array(data)
                data = tf.keras.backend.expand_dims(data, axis=0)
                data = tf.dtypes.cast(data, tf.float32)
                output = model(data)
                # loss = loss_object(target, output)
                output_characters.append(alphabet.get_char_by_vector(output))
            # compile the reconstructed word
            words_pred.append("".join(output_characters))
            # save the true word for the distance calculation
            words_true.append(str(cognate_set.ancestor_word))
        wts.extend(words_true)
        wps.extend(words_pred)

        # create plots
        ld = LevenshteinDistance(words_true, words_pred)
        ld.print_distances()
        ld.print_percentiles()

        print("***** Testing finished *****")

    # save results after last run
    outfile = plots_dir / "deep_test_{}{}{}.jpg".format(args.model, "_aligned" if aligned else "",
                                                                    "_ortho" if ortho else "")
    title = "Model [Test]:  {}{}{}\n 5 cross-validation folds"\
        .format(", " + args.model, ", aligned" if aligned else "", ", orthographic" if ortho else "")
    ld = LevenshteinDistance(wts, wps)
    plot_results(title=title,
                 distances={"=<" + str(d): count/5 for d, count in ld.distances.items()},
                 percentiles={"=<" + str(d): perc for d, perc in ld.percentiles.items()},
                 mean_dist=ld.mean_distance,
                 mean_dist_norm=ld.mean_distance_normalized,
                 losses=[],
                 outfile=Path(outfile),
                 testing=True)


if __name__ == '__main__':
    main()
