import tensorflow as tf
from model import Encoder

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# loss function
def loss_function(real, pred):
    """ wrapper for spare categorical"""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


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
        decoded_input = tf.expand_dims([target.word_index["<start>"]] * BAT, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predicted, decoded_hidden, _ = decoder(decoded_input, decoded_hidden, encoded_output)
            loss += loss_function(target[:, t], predicted, loss_object)
            # using teacher forcing
            decoded_input = tf.expand_dims(target[:, t], 1)

        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

