from pathlib import Path
import tensorflow as tf

from classes import Alphabet

ipa = Alphabet(Path("../data/alphabets/ipa.csv"))

chars = "abcdefghijklmnop"

for char in chars:
    vec = ipa.create_char(char).vector
    char_ = ipa.get_char_by_feature_vector(vec)
    print(char, char_)

vec = [0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
vec_ = []
for i in vec:
    if i == 0:
        vec_.append(0.1)
    elif i == 1:
        vec_.append(0.9)
print(ipa.get_char_by_feature_vector(vec_))

import numpy as np
from utils import create_model

model, optimizer, loss_object = create_model(len(vec), 64, 32, len(vec))
vec = tf.keras.backend.expand_dims(vec, axis=0)
vec = tf.dtypes.cast(vec, tf.float32)

losses = []
with tf.GradientTape(persistent=True) as tape:
    for i in range(20):
        output = model(vec)
        loss = loss_object(vec, output)
        losses.append(loss)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        print(i, float(loss), list(output))

char = ipa.get_char_by_feature_vector(output)
print(char)
