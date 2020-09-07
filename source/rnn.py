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

#sys.path.insert(0, "/home/julius/git/my_repos/sopro-nlpwithnn/")

from classes import Alphabet, CognateSet
#from source.net.model import Encoder, Decoder
#from source.net.train import loss_function, train_step

#set the weights to initialize to the same weight each time
tf.random.set_seed(seed=666)

def build_model(input_dim, embedding_dim, lstm_dim, output_dim, langs):
    input_layers = []
    lstm_layers = []
    for lang in langs:
        #print("lang")
        #print(lang)
        input_layer = tf.keras.Input(shape = (1,27), name=lang)
        #print("input_layer")
        #print(input_layer)
        lstm_layers.append(tf.keras.layers.LSTM(lstm_dim)(input_layer))
        #print("lstm layer")
        #print(lstm_layers)
        input_layers.append(input_layer)

    output = tf.keras.layers.concatenate(lstm_layers, axis=1)
    output = tf.keras.layers.Dense(output_dim, activation='relu', name='output')(output)
    model = tf.keras.models.Model(input_layers, [output])

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='cosine-similarity', optimizer=optimizer)
    return model, optimizer

#langs = ['italian', 'spanish', 'french', 'portuguese', 'romanian']
#trial_model = build_model(input_dim=27, embedding_dim=100, lstm_dim=32, output_dim=100, langs=langs)
#model, optimizer = trial_model
#print(model, optimizer)

model = keras.Sequential([
    layers.Embedding(input_dim=28, output_dim=100),
    layers.SimpleRNN(32), layers.Dense(28)])

# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
#model.add(layers.Embedding(input_dim=27, output_dim=100))

# Add a LSTM layer with 128 internal units.
#model.add(layers.SimpleRNN(32))

# Add a Dense layer with 10 units.
#model.add(layers.Dense(27))

#model.add(layers.Flatten())

#model.summary()
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4


HEADER_ROW = 0
COLUMN_SEPARATOR = ","
ID_COLUMN = 0
CONCEPT_COLUMN = 1
BATCH_SIZE = 1

embedding_dim = 10
vocab_input_size = 27

#encoder = Encoder(vocab_input_size, embedding_dim, encoded_units=10, batch_size=BATCH_SIZE)
#decoder = Decoder(vocab_input_size, embedding_dim, decoded_units=10, batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.SGD()
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4
#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss_object = tf.keras.losses.CosineSimilarity()

model.compile(loss="cosine_similarity", optimizer=optimizer)

#model.compile(loss="mean_squared_error", optimizer=optimizer)
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4
models = ['asjp', 'ipa', 'latin']

valid_size = 0.8
batch_size = 1

pad_to = 10  # also vocab size

output_characters = []
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/romance_ipa_full.csv",
                        help="file containing the cognate sets")
    parser.add_argument("--model", type=str, default="../data/alphabets/ipa.csv",
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4
                        help="file containing the character embeddings")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--model", type=str, default='latin', help="the model to be trained")
    parser.add_argument("--ancestor", type=str, default='latin', help="column corresponding to the proto-form")
    parser.add_argument("--orthographic", default=0, action='count',
                        help="switch between orthographic and feature-based approach")
    return parser.parse_args()

def vector_to_char(vector: np.array, alphabet:Alphabet):
    return alphabet.get_char_by_feature_vector(vector)


def main():
    args = parser_args()
    data_file = Path(args.data)
    assert data_file.exists() and data_file.is_file(), "Data file {} does not exist".format(data_file)
    alphabet_file = Path(args.alphabet)
    assert alphabet_file.exists() and alphabet_file.is_file(), "Alphabet file {} does not exist".format(alphabet_file)
    alphabet = Alphabet(alphabet_file, encoding='utf-16')
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4
    assert isinstance(args.epochs, int), "Epochs not int, but {}".format(type(args.epochs))
    assert args.epochs > 0, "Epochs out of range: {}".format(args.epochs)
    epochs = args.epochs
    assert args.model in models, "Model must be one of {}".format(models)
    #args.model
    print("alphabet:")
    print(alphabet)
    assert args.orthographic in [0, 1], "Too many instances of --orthographic switch, should be one of [0, 1]"
    orthographic = bool(args.orthographic)

    print("data_file: {}".format(data_file.absolute()))
    print("model: {}, orthographic={}".format(model, orthographic))
    print("alphabet: {}, read from {}".format(model, alphabet_file.absolute()))
    print("epochs: {}".format(epochs))

    # create cognate sets

    cognate_sets = []

    data = data_file.open(encoding='utf-16').read().split("\n")
    cols = data[HEADER_ROW].split(COLUMN_SEPARATOR)
    langs = cols[2:]

    for li, line in enumerate(data[HEADER_ROW:]):
        if line == "" or li % 2 != 0:
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4
            continue
        row_split = line.split(COLUMN_SEPARATOR)
        id = row_split[ID_COLUMN]
        concept = row_split[CONCEPT_COLUMN]
        words = row_split[CONCEPT_COLUMN + 1:]
        cognate_dict = {}
        assert len(langs) == len(words), "Langs / Words mismatch, expected {}, got {}".format(len(langs), len(words))
        for lang, word in zip(langs, words):
            cognate_dict[lang] = alphabet.translate(word)
        cs = CognateSet(id=id, concept=concept, ancestor='latin', cognate_dict=cognate_dict, alphabet=alphabet,
                        pad_to=pad_to)
        cognate_sets.append(cs)

    

>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4

    split_index = int(valid_size * len(cognate_sets))
    train_data = cognate_sets[:split_index]
    valid_data = cognate_sets[split_index:]
    print("train size: {}".format(len(train_data)))
    print("valid size: {}".format(len(valid_data)))


    for epoch in range(epochs):
        epoch_loss = []
        #initialize the GradientTape
        with tf.GradientTape(persistent=True) as tape:
            #iterate over the cognate sets
            for i, cs in enumerate(train_data):
                #print("cognate sets")
                print(cs)
                batch_loss = []
                #iterate over the character embeddings
                for j, char_embedding in enumerate(cs):
                   # print("j")
                   # print(j)
                   # print("char embedding")
                   # print(char_embedding)
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4
                    #add a dimension to the latin character embedding (ancestor embedding)
                    #we add a dimension because we use a batch size of 1 and TensorFlow does not
                    #automatically insert the batch size dimension
                    target = tf.keras.backend.expand_dims(char_embedding.pop(cs.ancestor).to_numpy(), axis=0)
                    #convert the latin character embedding to float32 to match the dtype of the output (line 137)
                    target = tf.dtypes.cast(target, tf.float32)
                    #iterate through the embeddings
                    for lang, embedding in char_embedding.items():
                      # print("lang")
                       # print(lang)
                        #print("embeddding")
                        #print(embedding)
                        #add a dimension to the the embeddings
                        data = tf.keras.backend.expand_dims(embedding.to_numpy(), axis=0)
                        output = model(data)
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4
                        #calculate the loss
                        loss = loss_object(target, output)
                        epoch_loss.append(float(loss))
                        batch_loss.append(float(loss))
                        #calculate the gradients
                        gradients = tape.gradient(loss, model.trainable_weights)
                        #backpropagate
                        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                    #print("output")
                    #print(output)
                    #convert the character vector into a character 
                    output_word = " "
                    output_word += vector_to_char(output, alphabet)
                    #print("output word")
                    #print(output_word)
                    #append the converted vectors to a list so we can see the reconstructed word
                    output_characters.append(output_word)
                   # print("output characters")
                   # print(output_characters)
                    #reset the output word string so we can create another reconstructed word
                    output_word = None
                #get the reconstructed word
                print("Reconstructed word={}".format(output_characters))
                print("Latin word: {}".format(cs.get_ancestor()))
                #clear the list of output characters so we can create another word
                output_characters.clear()     
                print("Batch {}, loss={}".format(i, np.mean(batch_loss)))
                print("Epoch {}, loss={}".format(epoch, np.mean(epoch_loss)))
>>>>>>> 14044efecef78889d7560cf204e25e0bbc20a2a4




if __name__ == '__main__':
    main()
