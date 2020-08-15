# source: https://www.tensorflow.org/tutorials/text/nmt_with_attention
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import io
import time
import os

from utils import load_dataset, preprocess_sentence
from model import Encoder, Decoder, BahdanauAttention
from train import train_step
from train import loss_function

print("Tensorflow version: ", tf.__version__)

BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024

# download data
path_to_zip = tf.keras.utils.get_file(
    "spa-eng.zip",
    origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True
)

path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"


# limit data for development
num_examples = 30000

input_tensor, input_language, target_tensor, target_language = load_dataset(path_to_file, num_examples)
max_length_target, max_length_input = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
    train_test_split(input_tensor, target_tensor, test_size=0.2)
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

# create tensorflow dataset
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_input_size = len(input_language.word_index) + 1
vocab_target_size = len(target_language.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print(example_input_batch.shape, example_target_batch.shape)

encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)
# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
# optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# checkpoints
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


@tf.function
def train_step(input, target, encoded_hidden):
    """
     performs one training step (on batch)
    :param input:
    :param target:
    :param target_lang:
    :param encoded_hidden:
    :param optimizer:
    :param encoder:
    :param decoder:
    :param batch_size:
    :return:
    """

    loss = 0

    with tf.GradientTape() as tape:
        encoded_output, encoded_hidden = encoder(input, encoded_hidden)
        decoded_hidden = encoded_hidden
        decoded_input = tf.expand_dims([target_language.word_index["<start>"]] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predicted, decoded_hidden, _ = decoder(decoded_input, decoded_hidden, encoded_output)

            loss += loss_function(target[:, t], predicted)
            # using teacher forcing
            decoded_input = tf.expand_dims(target[:, t], 1)

        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
# train
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  encoded_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
      batch_loss = train_step(input, target, encoded_hidden)
      total_loss += batch_loss

      if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix=checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

def evaluate(sentence):
  attention_plot = np.zeros((max_length_target, max_length_input))

  sentence = preprocess_sentence(sentence)

  inputs = [input_language.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_input,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([target_language.word_index['<start>']], 0)

  for t in range(max_length_target):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += target_language.index_word[predicted_id] + ' '

    if target_language.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  #plot_attention(attention_plot, sentence.split(' '), result.split(' '))