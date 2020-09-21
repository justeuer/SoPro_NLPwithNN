import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from argparse import ArgumentParser
from pathlib import Path
# from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import sys

from classes import Alphabet, CognateSet, LevenshteinDistance
from utils import create_lstm_model
from plots import plot_results
import nltk
import os


plots_dir = Path("../out/plots_ciobanu_lstm")
if not plots_dir.exists():
    plots_dir.mkdir(parents=True)

results_dir = Path("../out/results")

#save the model
checkpoint_path = Path("../ciobanu_lstm_model/epochs/")
if not checkpoint_path.exists():
	checkpoint_path.mkdir(parents=True)
#checkpoint_dir = os.path.dirname(checkpoint_path)

# create output dir
# out_dir = Path("../data/out/ciobanu")
# if not out_dir.exists():
#    out_dir.mkdir()

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
                        default="../data/ciobanu/romance_asjp_auto.csv",
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
                        default=2,
                        help="number of epochs")
    parser.add_argument("--out_tag",
                        type=str,
                        default="swadesh",
                        help="tag for output directories")
    return parser.parse_args()


def train():
    
    # Command line call I used:
    # python ciobanu_rnn.py --data=ipa --model=ipa --epochs=10 --out_tag=test --model=ipa --ancestor=ancestor

    global encoding
    args = parser_args()
    # determine whether the model should use feature encodings or character embeddings
    assert args.ortho in [0, 1], "Too many instances of --orthographic switch, should be 0 or 1"
    ortho = bool(args.ortho)
    # determine whether to use the aligned or unaligned data
    assert args.aligned in [0, 1], "Too many instances of --aligned switch, should be 0 or 1"
    aligned = bool(args.aligned)
    # load data
    data_file = None
    if args.data == "ipa":
        encoding = 'utf-16'
        data_file = Path("../data/ciobanu/romance_ipa_auto.csv")
    elif args.data == "asjp":
        encoding = 'ascii'
        data_file = Path("../data/ciobanu/romance_asjp_auto.csv")
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
    alphabet = Alphabet(alphabet_file, encoding=encoding, ortho=ortho)
    assert isinstance(args.epochs, int), "Epochs not int, but {}".format(type(args.epochs))
    assert args.epochs > 0, "Epochs out of range: {}".format(args.epochs)
    epochs = args.epochs

    # ancestor
    ancestor = args.ancestor

    # determine output directories, create them if they do not exist
    out_tag = "_{}".format(args.out_tag)
    plots_dir = Path("../out/plots{}_lstm".format(out_tag))
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True)
    results_dir = Path("../out/results{}_lstm".format(out_tag))
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    # create file for results
    result_file_path = results_dir / "lstm_{}{}{}.txt".format(args.model,
                                                              "_aligned" if aligned else "",
                                                              "_ortho" if ortho else "")
    result_file_path.touch()
    result_file = result_file_path.open('w', encoding=encoding)

    print("alphabet:")
    print(alphabet)

    # initialize model
    model, optimizer, loss_object = create_lstm_model(input_dim=alphabet.get_feature_dim(),
                                                 embedding_dim=28,
                                                 context_dim=128,
                                                 output_dim=alphabet.get_feature_dim())

    model.summary()

    print("data_file: {}".format(data_file.absolute()))
    print("model: {}, orthographic={}, aligned={}".format(args.model, ortho, aligned))
    print("alphabet: {}, read from {}".format(args.model, alphabet_file.absolute()))
    print("epochs: {}".format(epochs))

    # create cognate sets

    cognate_sets = []

    data = data_file.open(encoding='utf-16').read().split("\n")
    cols = data[HEADER_ROW].split(COLUMN_SEPARATOR)
    langs = cols[2:]
    print("langs")
    print(langs)

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
        # print("words")
        # print(words)
        cognate_dict = {}
        assert len(langs) == len(words), "Langs / Words mismatch, expected {}, got {}".format(len(langs), len(words))
        for lang, word in zip(langs, words):
            # print("lang, word")
            # print(lang, word)
            cognate_dict[lang] = alphabet.translate(word)
        cs = CognateSet(id=id,
                        concept=concept,
                        ancestor=ancestor,
                        cognate_dict=cognate_dict,
                        alphabet=alphabet)
        cognate_sets.append(cs)

    # maybe we needn't do the evaluation, since we mainly want to know how
    # the model behaves with the different inputs

    split_index = int(valid_size * len(cognate_sets))
    train_data = cognate_sets[:split_index]
    valid_data = cognate_sets[split_index:]
    print("train size: {}".format(len(train_data)))
    print("valid size: {}".format(len(valid_data)))
    cognate_sets = cognate_sets[:50]
    # print("cognate_sets in ral")
    # print(cognate_sets)

    words_true = []
    words_pred = []
    epoch_losses = []
    batch_losses = []

    for epoch in range(1, epochs + 1):
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
                       # print("output")
                        #print(output)
                        # calculate the loss
                        loss = loss_object(target, output)
                        epoch_losses.append(float(loss))
                        batch_losses.append(float(loss))
                        # calculate the gradients
                        gradients = tape.gradient(loss, model.trainable_weights)
                        # backpropagate
                        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                        model.save_weights("../ciobanu_lstm_model/epochs/epoch_{}.hd5".format(epoch))
                        # convert the character vector into a character
                    output_char = alphabet.get_char_by_feature_vector(output)
                    # append the converted vectors to a list so we can see the reconstructed word
                    output_characters.append(output_char)
            # append the reconstructed word and the ancestor to the true/pred lists
            words_pred.append("".join(output_characters))
            #print("predicted words")
           # print(words_pred)
            words_true.append(str(cs.ancestor))
            #print("true words")
           # print(words_true)
            if i % 100 == 0:
                print("Epoch [{}/{}], Batch [{}/{}]".format(epoch, epochs, i, len(cognate_sets)))
            # clear the list of output characters so we can create another word
            output_characters.clear()
            #print("Batch {}, mean loss={}".format(i, np.mean(batch_losses)))
        # calculate distances
        ld = LevenshteinDistance(true=words_true,
                                 pred=words_pred)
        print("Epoch {} finished".format(epoch))
        print("Mean loss={}".format(epoch, np.mean(epoch_losses)))
        ld.print_distances()
        ld.print_percentiles()
        print("epochs are finished")
        print(epoch)
        if epoch == epochs:
        	print("this is the last epoch")
        	print(epoch)
        	# save reconstructed words (but only if the edit distance is at least one)
        	outfile = "../out/plots_ciobanu_lstm/lstm_{}{}{}.jpg".format(args.model, "_aligned" if aligned else "", "_ortho" if ortho else "")
        	title = "Model: lstm net{}{}{}".format(", " + args.model, ", aligned" if aligned else "", ", orthographic" if ortho else "")
        	plot_results(title=title,
                         distances={"=<" + str(d): count for d, count in ld.distances.items()},
                         percentiles={"=<" + str(d): perc for d, perc in ld.percentiles.items()},
                         mean_dist=ld.mean_distance,
                         mean_dist_norm=ld.mean_distance_normalized,
                         losses=epoch_losses,
                         outfile=Path(outfile))
        	# save reconstructed words (but only if the edit distance is at least one)
        	for t, p in zip(words_true, words_pred):
        		distance = nltk.edit_distance(t, p)
        		if distance > 0:
        			line = "{},{},distance={}\n".format(t, p, nltk.edit_distance(t, p))
        			result_file.write(line)
        	result_file.close()


if __name__ == '__main__':
    train()