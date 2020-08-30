from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import sys

sys.path.insert(0, "/home/julius/git/my_repos/sopro-nlpwithnn/")

from source.classes import Alphabet, CognateSet
from source.net.model import Encoder, Decoder
from source.net.train import loss_function, train_step

HEADER_ROW = 0
COLUMN_SEPARATOR = ","
ID_COLUMN = 0
CONCEPT_COLUMN = 1
BATCH_SIZE = 1

embedding_dim = 10
vocab_input_size = 26

encoder = Encoder(vocab_input_size, embedding_dim, encoded_units=10, batch_size=BATCH_SIZE)
decoder = Decoder(vocab_input_size, embedding_dim, decoded_units=10, batch_size=BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#loss_object = tf.keras.losses.CosineSimilarity()
models = ['asjp', 'ipa', 'latin']

valid_size = 0.8
batch_size = 1
pad_to = 6  # also vocab size


def parser_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="../../data/romance_asjp_full.csv",
                        help="file containing the cognate sets")
    parser.add_argument("--alphabet", type=str, default="../../data/alphabets/asjp.csv",
                        help="file containing the character embeddings")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--model", type=str, default='latin', help="the model to be trained")
    parser.add_argument("--ancestor", type=str, default='latin', help="column corresponding to the proto-form")
    parser.add_argument("--orthographic", default=0, action='count',
                        help="switch between orthographic and feature-based approach")
    return parser.parse_args()


def main():
    args = parser_args()
    data_file = Path(args.data)
    assert data_file.exists() and data_file.is_file(), "Data file {} does not exist".format(data_file)
    alphabet_file = Path(args.alphabet)
    assert alphabet_file.exists() and alphabet_file.is_file(), "Alphabet file {} does not exist".format(alphabet_file)
    alphabet = Alphabet(alphabet_file, encoding='ascii')
    assert isinstance(args.epochs, int), "Epochs not int, but {}".format(type(args.epochs))
    assert args.epochs > 0, "Epochs out of range: {}".format(args.epochs)
    epochs = args.epochs
    assert args.model in models, "Model must be one of {}".format(models)
    model = args.model
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
        cs = CognateSet(id=id, concept=concept, ancestor='latin', cognate_dict=cognate_dict, alphabet=alphabet,
                        pad_to=pad_to)
        cognate_sets.append(cs)

    split_index = int(valid_size * len(cognate_sets))
    train_data = cognate_sets[:split_index]
    valid_data = cognate_sets[split_index:]
    print("train size: {}".format(len(train_data)))
    print("valid size: {}".format(len(valid_data)))

    for epoch in range(epochs):
        epoch_start = time.time()
        encoded_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for i, cs in enumerate(train_data):
            print(cs)
            for j, char_embedding in enumerate(cs):
                # print(char_embedding)
                target = char_embedding.pop(cs.ancestor)
                data = char_embedding
                print("target (latin)", target.to_numpy())
                for lang, embedding in data.items():
                    print("embedding ({})".format(lang), embedding.to_numpy())
                    batch_loss = train_step(embedding.to_numpy(), target, encoder, decoder, encoded_hidden)
                    total_loss += batch_loss
                if i % 100:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                     i,
                                                                     batch_loss.numpy()))
                #print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                #                                            total_loss / steps_per_epoch))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - epoch_start))

        # TODO: evaluate


if __name__ == '__main__':
    main()
