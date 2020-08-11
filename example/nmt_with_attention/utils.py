import unicodedata
import re
import io
import tensorflow as tf


def unicode_to_ascii(s: str):
    """
    converts a unicode string to ascii
    :param s:
    :return:
    """

    return ''.join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(w):
    """
    convert sentence into form "<start> w1 ... wn . <end>, also replacing numeric symbols"
    :param sent:
    :return:
    """

    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples):
    """
    creates a bilingual dataset
    :param path:
    :param num_examples:
    :return:
    """

    lines = io.open(path, encoding="UTF-8").read().strip().split('\n')
    word_pairs = [
        [preprocess_sentence(w) for w in l.split('\t')]
        for l in lines[:num_examples]
    ]

    return zip(*word_pairs)


def tokenize(language):
    """
    tokenizes sentences of a language, pad sentences to uniform length
    :param language:
    :return:
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(language)
    tensor = tokenizer.texts_to_sequences(language)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer


def load_dataset(path, num_examples=None):
    """
    loads datasets and transform to tensors
    :param path:
    :param num_examples:
    :return:
    """
    target_language, input_language = create_dataset(path, num_examples)
    input_tensor, input_tokenizer = tokenize(input_language)
    target_tensor, target_tokenizer = tokenize(target_language)

    return input_tensor, input_tokenizer, target_tensor, target_tokenizer


def convert(lang, tensor):
    """
    converts word in sentence tensor to word in language
    :param lang:
    :param tensor:
    :return:
    """
    for t in tensor:
        if t != 0:
            print(t)
            print("{}\t-->\t{}".format(t, lang.index_word[t]))

