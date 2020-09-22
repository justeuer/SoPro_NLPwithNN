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
from utils import create_model
from plots import plot_results
import nltk
import os
import random

encoding = None
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
valid_size = 0.8


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
                        default=10,
                        help="number of epochs")
    parser.add_argument("--out_tag",
                        type=str,
                        default="swadesh",
                        help="tag for output directories")
    return parser.parse_args()


def main():
    
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
    elif args.data == "latin":
        encoding = 'utf-16'
        data_file = Path("../data/ciobanu/romance_orthographic.csv")
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
    elif args.model == "latin":
        encoding = 'utf-16'
        alphabet_file = Path("../data/alphabets/latin.csv")
    # load data from file
    assert alphabet_file.exists() and alphabet_file.is_file(), "Alphabet file {} does not exist".format(alphabet_file)
    alphabet = Alphabet(alphabet_file, encoding=encoding, ortho=ortho)
    
    #number of epochs
    assert isinstance(args.epochs, int), "Epochs not int, but {}".format(type(args.epochs))
    assert args.epochs > 0, "Epochs out of range: {}".format(args.epochs)
    epochs = args.epochs


    # determine output directories, create them if they do not exist
    out_tag = "_{}".format(args.out_tag)
    plots_dir = Path("../out/plots{}_rnn".format(out_tag))
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True)
    results_dir = Path("../out/results{}_rnn".format(out_tag))
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    # create file for results
    result_file_path = results_dir / "rnn{}{}{}.txt".format(args.model,
                                                              "_aligned" if aligned else "",
                                                              "_ortho" if ortho else "")
    result_file_path.touch()
    result_file = result_file_path.open('w', encoding=encoding)

    #determine ancestor
    ancestor = args.ancestor


    # create cognate sets

    cognate_sets = []

    data = data_file.open(encoding='utf-16').read().split("\n")
    cols = data[HEADER_ROW].split(COLUMN_SEPARATOR)
    langs = cols[2:]
    

    for li, line in enumerate(data[HEADER_ROW:]):
        if args.model == 'latin':
            if line == "":
                continue
        elif aligned:
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
    
    # prepare train_test_split
    data = {i: cognate_set for i, cognate_set in enumerate(cognate_sets)}
    train_size = 0.8
    n_train_samples = int(train_size * len(cognate_sets))
    train_indices = random.sample(list(data), n_train_samples)
    train_data = {i: cognate_set for i, cognate_set in data.items() if i in train_indices}
    test_data = {i: cognate_set for i, cognate_set in data.items() if i not in train_indices}
    print("Train size: {}".format(len(train_data)))
    print("Test size: {}".format(len(test_data)))


    # initialize model
    model, optimizer, loss_object = create_model(input_dim=alphabet.get_feature_dim(),
                                                 embedding_dim=28,
                                                 context_dim=128,
                                                 output_dim=alphabet.get_feature_dim())

    model.summary()



    words_true = []
    words_pred = []
    epoch_losses = []
    batch_losses = []

    print("*** Start training ***")
    for epoch in range(1, epochs + 1):
        # reset lists
        words_true.clear()
        words_pred.clear()
        batch_losses.clear()
        # iterate over the cognate sets
        for batch, cognate_set in train_data.items():
            output_characters = []
            # iterate over the character embeddings
            for lang_array in cognate_set:
                # add a dimension to the latin character embedding (ancestor embedding)
                # we add a dimension because we use a batch size of 1 and TensorFlow does not
                # automatically insert the batch size dimension
                target = tf.keras.backend.expand_dims(lang_array.pop(ancestor).to_numpy(), axis=0)
                # convert the latin character embedding to float32 to match the dtype of the output (line 137)
                target = tf.dtypes.cast(target, tf.float32)
                data = []
                for lang, vec in lang_array.items():
                    data.append(list(vec))
                data = np.array(data)
                data = tf.keras.backend.expand_dims(data, axis=0)
                data = tf.dtypes.cast(target, tf.float32)
                # iterate through the embeddings
                # initialize the GradientTape
                with tf.GradientTape() as tape:
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
                    #output_char = alphabet.get_char_by_feature_vector(output)
                    # append the converted vectors to a list so we can see the reconstructed word
                    output_characters.append(alphabet.get_char_by_vector(output))
                  #  print("output_characters")
                  #  print(output_characters)
            # append the reconstructed word and the ancestor to the true/pred lists
            words_pred.append("".join(output_characters))
           # print("predicted words")
           # print(words_pred)
            words_true.append(str(cs.ancestor_word))
           # print("true words")
           # print(words_true)
            if batch % 100 == 0:
                print("Epoch [{}/{}], Batch [{}/{}]".format(epoch, epochs, batch, len(cognate_sets)))
            # clear the list of output characters so we can create another word
            #output_characters.clear()
            #print("Batch {}, mean loss={}".format(i, np.mean(batch_losses)))
       
        #calculate mean epoch loss
        mean_loss = np.mean(batch_losses)
        epoch_losses.append(mean_loss)
        print("Epoch[{}]/[{}], mean batch loss = {}".format(epoch, epochs, mean_loss))

        # calculate distances
        ld = LevenshteinDistance(true=words_true,
                                 pred=words_pred)
        ld.print_distances()
        ld.print_percentiles()
        print("epochs are finished")
        print(epoch)
        if epoch == epochs:
        	print("this is the last epoch")
        	print(epoch)
        	# save reconstructed words (but only if the edit distance is at least one)
        	outfile = plots_dir / "ciobanu_rnn_{}{}{}.jpg".format(args.model, "_aligned" if aligned else "", "_ortho" if ortho else "")
        	title = "Model: ciobanu rnn{}{}{}".format(", " + args.model, ", aligned" if aligned else "", ", orthographic" if ortho else "")
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
    words_pred.clear()
    words_true.clear()
    print("***** Training finished *****")
    print()


    print("***** Start testing *****")
    for i, cognate_set in test_data.items():
        output_characters = []
        for lang_array in cognate_set:
            target = tf.keras.backend.expand_dims(lang_array.pop(ancestor).to_numpy(), axis=0)
            target = tf.dtypes.cast(target, tf.float32)
            data = []
            for lang, vec in lang_array.items():
                data = np.array(vec)
            data = tf.keras.backend.expand_dims(data, axis=0)
            data = tf.dtypes.cast(data, tf.float32)
            #print("testing data")
            #print(data)
            output = model(data)
            # loss = loss_object(target, output)
            output_characters.append(alphabet.get_char_by_vector(output))
        words_pred.append("".join(output_characters))
        words_true.append(str(cognate_set.ancestor_word))

    # create plots
    ld = LevenshteinDistance(words_true, words_pred)
    ld.print_distances()
    ld.print_percentiles()
    outfile = plots_dir / "ciobanu_rnn_{}{}{}.jpg".format(args.model, "_aligned" if aligned else "",
                                                            "_ortho" if ortho else "")
    title = "Model [Test]: ciobanu rnn, {}{}{}".format(", " + args.model, ", aligned" if aligned else "",
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