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

from utils import load_dataset
from model import Encoder, Decoder
from train import train_step

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
max_length_input, max_length_target = input_tensor.shape[1], target_tensor.shape[0]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
    train_test_split(input_tensor, target_tensor, test_size=0.2)
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

# create tensorflow dataset
BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE
V_input = len(input_language.word_index) + 1
V_target = len(target_language.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(V_input, EMBEDDING_DIM, UNITS, BATCH_SIZE)
decoder = Decoder(V_target, EMBEDDING_DIM, UNITS, BATCH_SIZE)
# optimizer
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# checkpoints
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# train
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  encoded_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (input, target)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(input, target, target_language, encoded_hidden, optimizer, encoder, decoder, BATCH_SIZE, loss_object)
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
