import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=26, output_dim=10))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(32))

# Add a Dense layer with 10 units.
model.add(layers.Dense(26))

model.summary()