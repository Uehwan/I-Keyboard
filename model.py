from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, GRU

import time
import numpy as np
import matplotlib.pyplot as plt

from utils import positional_encoding, EncoderLayer


class TransformerDecoder(tf.keras.layers.Layer):
    """Decoder based on a transformer encoder
    Though transformer consists of a pair of encoder and decoder,
    we employ just the encoder part for semantic decoding
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, rate=0.1):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class BiRNNDecoder(tf.keras.layers.Layer):
    """Decoder based on Bi-RNN (GRU)
    Could be both statistical and semantic decoders of I-Keyboard
    """
    def __init__(self, rnn_units, vocab_size):
        super(BiRNNDecoder, self).__init__()
        self.model = tf.keras.Sequential([
            Bidirectional(
                GRU(rnn_units, return_sequences=True, stateful=True,
                    recurrent_initializer='glorot_uniform')
            ),
            Bidirectional(
                GRU(rnn_units, return_sequences=True, stateful=True,
                    recurrent_initializer='glorot_uniform')
            ),
            tf.keras.layers.Dense(vocab_size)
        ])

    def call(self, x):
        return self.model(x)


class DeepNeuralDecoder(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, x):
        pass


class TMIDecoderLongTerm(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, x):
        pass


class TMIDecoderShortTerm(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def call(self, x):
        pass


if __name__ == "__main__":
    print("#############################################")
    print("Test Encoder")
    print("#############################################")
    sample_trans_decoder = TransformerDecoder(num_layers=2, d_model=512, num_heads=8, 
        dff=2048, input_vocab_size=8500, maximum_position_encoding=10000)
    sample_trans_decoder_output = sample_trans_decoder(
        tf.random.uniform((64, 62)), training=False, mask=None)
    print(sample_trans_decoder_output.shape)  # (batch_size, input_seq_len, d_model)

    print("#############################################")
    print("Test BiRNNDecoder")
    print("#############################################")
    input_vec_dim = 17
    sample_birnn_decoder = BiRNNDecoder(rnn_units=64, vocab_size=16)
    sample_birnn_decoder_output = sample_birnn_decoder(
        tf.random.uniform((32, 47, input_vec_dim))
    )
    print(sample_birnn_decoder_output.shape)  # (batch_size, input_seq_len, vocab_size)
    sample_indices = tf.random.categorical(sample_birnn_decoder_output[0], num_samples=1)
    sample_indices = tf.squeeze(sample_indices, axis=-1).numpy()
    print(sample_indices.shape)
    sample_preds = tf.argmax(
        tf.keras.activations.softmax(sample_birnn_decoder_output, axis=-1),
        axis=-1
    )
