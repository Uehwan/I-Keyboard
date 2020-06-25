import tensorflow as tf
from modules import *
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.layers import core as layers_core
import pickle

BP = pickle.load(open('./data_processor_aug.pkl', 'rb'))


class BaseModel(object):
    def __init__(self, inputs, gs, option):
        target = inputs['id_seq']
        x_input = inputs['x_seq']
        y_input = inputs['y_seq']
        lengths = inputs['length']

        self.option = option
        self.lr = tf.placeholder(shape=None, dtype=tf.float32)
        # self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)
        self.gs = gs
        self.increment_gs = tf.assign(self.gs, self.gs + 1)  # To increment during val

        with tf.variable_scope("main", initializer=xavier_initializer()):
            loss, preds = self.get_model_loss(lengths, x_input, y_input, target)

            self.preds = preds
            self.loss = tf.reduce_mean(loss)
            opt = tf.train.AdamOptimizer(self.lr)
            self.train = opt.minimize(self.loss, global_step=self.gs)
            self.target = target
            self.x_input, self.y_input = x_input, y_input
            self.lengths = lengths
            self.accuracy = tf.contrib.metrics.accuracy(predictions=self.preds, labels=self.target)
            self.write_op = None
            self.make_summaries(loss)

    def make_summaries(self, loss):
        """
        Some summaries for Tensorflow
        """
        tf.summary.scalar("batch_accuracy", self.accuracy)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("lr", self.lr)
        # tf.summary.scalar("keep_prob", self.keep_prob)
        self.write_op = tf.summary.merge_all()

    def run_rnn(self, lengths, x_seq, y_seq):
        """
        Get a sequence, embed it and then run it through a GRU.
        We pass the lengths of the sequence to the dynamic rnn.
        We return the outputs for the language model and the state for the prediction of the book
        """
        x_expanded, y_expanded = tf.expand_dims(x_seq, 2), tf.expand_dims(y_seq, 2)
        source = tf.to_float(tf.concat([x_expanded, y_expanded], 2))
        if self.option.rnn_type == "LSTM":
            self.cell = tf.contrib.rnn.LSTMBlockCell
        else:
            self.cell = tf.contrib.rnn.GRUCell

        if self.option.model == "unidirectional":
            layers = [self.cell(self.option.num_unit) for _ in range(self.option.num_layer)]
            cell_in = tf.nn.rnn_cell.MultiRNNCell(layers)
            outputs, _ = tf.nn.dynamic_rnn(
                cell=cell_in,
                inputs=source,
                sequence_length=lengths,
                dtype=tf.float32
            )
        else:
            cells_fw = tf.contrib.rnn.MultiRNNCell([self.cell(self.option.num_unit)
                                                    for _ in range(self.option.num_layer)])
            cells_bw = tf.contrib.rnn.MultiRNNCell([self.cell(self.option.num_unit)
                                                    for _ in range(self.option.num_layer)])
            outputs_raw, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells_fw,
                cell_bw=cells_bw,
                inputs=source,
                sequence_length=lengths,
                dtype=tf.float32
            )
            output_fw, output_bw = outputs_raw
            outputs = tf.concat([output_fw, output_bw], 2)
        
        return outputs

    def get_model_loss(self, lengths, x_input, y_input, target):
        """
        We want the model to learn to predict the next character.
        We shift the input sequence one right and trim the right end of the outputs.
        So if sequence was
        ABCED ==> BCED
        And outputs (the GRU outputs)
        12345 ==> 1234
        Where we'd like to have the model learn that 1=B,2=C etc
        """
        # extra-layer
        outputs = self.run_rnn(lengths, x_input, y_input)
        logits = tf.contrib.layers.fully_connected(outputs, num_outputs=len(BP.vocab), activation_fn=None)
        mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
        preds = tf.argmax(tf.nn.softmax(logits), axis=2)
        loss = tf.losses.sparse_softmax_cross_entropy(target, logits, weights=mask)
        return loss, preds


class HierarchModel(BaseModel):
    def get_model_loss(self, lengths, x_input, y_input, target):
        """
        We want the model to learn to predict the next character.
        We shift the input sequence one right and trim the right end of the outputs.
        So if sequence was
        ABCED ==> BCED
        And outputs (the GRU outputs)
        12345 ==> 1234
        Where we'd like to have the model learn that 1=B,2=C etc
        """
        # extra-layer
        first_outputs, second_outputs = self.run_rnn(lengths, x_input, y_input)
        logits = tf.contrib.layers.fully_connected(second_outputs, num_outputs=len(BP.vocab), activation_fn=None)
        mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
        preds = tf.argmax(tf.nn.softmax(logits), axis=2)
        loss = tf.losses.sparse_softmax_cross_entropy(target, logits, weights=mask)
        if self.option.aux_sup:
            with tf.variable_scope("middle_layer", reuse=True):
                middle_logits = tf.contrib.layers.fully_connected(
                    first_outputs,
                    num_outputs=len(BP.vocab),
                    activation_fn=None
                )
            loss += tf.losses.sparse_softmax_cross_entropy(target, middle_logits, weights=mask)
        return loss, preds

    def run_rnn(self, lengths, x_seq, y_seq):
        """
        Get a sequence, embed it and then run it through a GRU.
        We pass the lengths of the sequence to the dynamic rnn.
        We return the outputs for the language model and the state for the prediction of the book
        """
        first_layer_outputs = super(HierarchModel, self).run_rnn(lengths, x_seq, y_seq)

        if self.option.embedding:
            with tf.variable_scope("middle_layer"):
                middle_logits = tf.contrib.layers.fully_connected(first_layer_outputs, num_outputs=len(BP.vocab),
                                                                  activation_fn=None)
                middle_preds = tf.argmax(tf.nn.softmax(middle_logits), axis=2)
                embedding_matrix = tf.get_variable("embedding", [len(BP.vocab), self.option.embedding_size])
                embedded_input = tf.nn.embedding_lookup(embedding_matrix, middle_preds)
        else:
            embedded_input = tf.identity(first_layer_outputs)

        with tf.variable_scope("second_layer"):
            cells_fw = tf.contrib.rnn.MultiRNNCell([self.cell(self.option.num_unit)
                                                    for _ in range(self.option.num_layer)])
            cells_bw = tf.contrib.rnn.MultiRNNCell([self.cell(self.option.num_unit)
                                                    for _ in range(self.option.num_layer)])
            outputs_raw, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells_fw,
                cell_bw=cells_bw,
                inputs=embedded_input,
                sequence_length=lengths,
                dtype=tf.float32
            )
            output_fw, output_bw = outputs_raw
            second_layer_outputs = tf.concat([output_fw, output_bw], 2)
        return first_layer_outputs, second_layer_outputs


class AttnModel(BaseModel):
    def __init__(self, inputs, gs, option):
        target = inputs['id_seq']
        x_input = inputs['x_seq']
        y_input = inputs['y_seq']
        lengths = inputs['length']

        self.option = option
        self.time_major = False

        self.lr = tf.placeholder(shape=None, dtype=tf.float32)
        self.gs = gs
        self.increment_gs = tf.assign(self.gs, self.gs + 1)  # To increment during val
        self.target = target
        # self.trunc_target = target[:, 1:]
        self.lengths = lengths
        # self.trunc_lengths = lengths - 1
        self.inference = None

        with tf.variable_scope("main", initializer=xavier_initializer()):
            loss, preds = self.run_rnn(lengths, x_input, y_input, target)

            self.preds = preds
            self.loss = tf.reduce_mean(loss)
            opt = tf.train.AdamOptimizer(self.lr)
            self.train = opt.minimize(self.loss, global_step=self.gs)
            self.x_input, self.y_input = x_input, y_input

            self.accuracy = tf.contrib.metrics.accuracy(predictions=self.preds, labels=self.target[:, 1:])
            self.write_op = None
            self.make_summaries(loss)

    def run_rnn(self, lengths, x_seq, y_seq, target):
        """
        Get a sequence, embed it and then run it through a GRU.
        We pass the lengths of the sequence to the dynamic rnn.
        We return the outputs for the language model and the state for the prediction of the book
        """
        lengths, x_seq, y_seq, target = tf.cast(lengths, tf.int32), tf.cast(x_seq, tf.int32), tf.cast(y_seq, tf.int32), tf.cast(target, tf.int32)
        x_expanded, y_expanded = tf.expand_dims(x_seq, -1), tf.expand_dims(y_seq, -1)
        source = tf.to_float(tf.concat([x_expanded, y_expanded], -1))
        if self.time_major:
            source = tf.transpose(source, perm=[1, 0, 2])

        # Encoder part
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        if self.option.rnn_type == "LSTM":
            cell = tf.contrib.rnn.LSTMBlockCell
        else:
            cell = tf.contrib.rnn.GRUCell

        '''
        cells_fw = tf.contrib.rnn.MultiRNNCell([cell(32) for _ in range(2)])
        cells_bw = tf.contrib.rnn.MultiRNNCell([cell(32) for _ in range(2)])

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cells_fw,
            cell_bw=cells_bw,
            inputs=source,
            sequence_length=lengths,
            dtype=tf.float32
        )
        output_fw, output_bw = outputs
        concat_output = tf.concat([output_fw, output_bw], 2)
        '''

        layers = [cell(self.option.num_unit) for _ in range(self.option.num_layer)]
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
            cell=encoder_cell,
            inputs=source,
            sequence_length=lengths,
            dtype=tf.float32,
            time_major=self.time_major
        )

        encoder_outputs = tf.contrib.layers.fully_connected(
            encoder_outputs,
            num_outputs=len(BP.vocab),
            activation_fn=None
        )

        # attention score
        if self.time_major:
            attention_states = tf.transpose(encoder_outputs, [1, 0, 2])  # [batch_size, max_time, num_units]
        else:
            attention_states = encoder_outputs
        attention = tf.contrib.seq2seq.LuongAttention(self.option.num_unit, attention_states,
                                                      memory_sequence_length=lengths)

        # Decoder part for training
        if self.option.embedding:
            embedding_decoder = tf.get_variable("embedding_decoder", [len(BP.vocab), self.option.embedding_size])
            decoder_embedded_input = tf.nn.embedding_lookup(embedding_decoder, target[:, :-1])
            if self.time_major:
                decoder_embedded_input = tf.transpose(decoder_embedded_input, [1, 0, 2])
        else:
            # without embedding
            decoder_embedded_input = tf.cast(tf.expand_dims(tf.transpose(target[:, :-1]), -1), tf.float32)

        decoder_layers = [cell(self.option.num_unit) for _ in range(self.option.num_layer)]
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layers)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention,
                                                           attention_layer_size=self.option.embedding_size)
        initial_state = decoder_cell.zero_state(self.option.batch_size,
                                                dtype=tf.float32).clone(cell_state=encoder_states)
        projection_layer = layers_core.Dense(len(BP.vocab), use_bias=False)

        preds, loss = self.decoder_training(
            decoder_embedded_input,
            decoder_cell,
            initial_state,
            projection_layer,
            lengths,
            target
        )

        if self.option.embedding:
            inference = self.decoder_inference(
                decoder_cell,
                initial_state,
                projection_layer,
                lengths,
                embedding_decoder
            )
        else:
            inference = self.decoder_inference(
                decoder_cell,
                initial_state,
                projection_layer,
                lengths,
            )
        self.inference = inference

        return loss, preds

    def decoder_training(self, decoder_input, decoder_cell,
                         initial_state, projection_layer, lengths, target):
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, lengths - 1, time_major=self.time_major)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer=projection_layer)

        # Dynamic decoding
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output
        mask = tf.sequence_mask(lengths - 1, tf.reduce_max(lengths - 1))
        preds = tf.argmax(tf.nn.softmax(logits), axis=2)
        loss = tf.losses.sparse_softmax_cross_entropy(target[:, 1:], logits, weights=mask)
        return preds, loss

    def decoder_inference(self, decoder_cell, initial_state, projection_layer, lengths, embedding_decoder=None):
        if self.option.embedding:
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding_decoder,
                tf.fill([self.option.batch_size], 0),
                end_token=1)
        else:
            return None

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            inference_helper,
            initial_state,
            output_layer=projection_layer)
        maximum_iterations = tf.round(tf.reduce_max(lengths - 1) * 2)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=maximum_iterations)
        # logits = outputs.rnn_output
        # inference = tf.argmax(tf.nn.softmax(logits), axis=2)
        return outputs.sample_id


class TransModel(BaseModel):
    def get_model_loss(self, lengths, x_input, y_input, target):
        # extra-layer
        outputs = self.run_rnn(lengths, x_input, y_input)
        logits = tf.contrib.layers.fully_connected(outputs, num_outputs=len(BP.vocab), activation_fn=None)
        mask = tf.sequence_mask(lengths, tf.reduce_max(lengths))
        preds = tf.argmax(tf.nn.softmax(logits), axis=2)
        loss = tf.losses.sparse_softmax_cross_entropy(target, logits, weights=mask)
        '''
        if self.option.aux_sup:
            with tf.variable_scope("middle_layer", reuse=True):
                middle_logits = tf.contrib.layers.fully_connected(
                    first_outputs,
                    num_outputs=len(BP.vocab),
                    activation_fn=None
                )
            loss += tf.losses.sparse_softmax_cross_entropy(target, middle_logits, weights=mask)
        '''
        return loss, preds

    def run_rnn(self, lengths, x_seq, y_seq):
        lengths, x_seq, y_seq = tf.cast(lengths, tf.int32), tf.cast(x_seq, tf.int32), tf.cast(y_seq, tf.int32)
        x_expanded, y_expanded = tf.expand_dims(x_seq, -1), tf.expand_dims(y_seq, -1)

        self.enc = tf.to_float(tf.concat([x_expanded, y_expanded], -1))
        self.enc = tf.contrib.layers.fully_connected(self.enc, num_outputs=self.option.num_unit, activation_fn=None)

        key_masks = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(self.enc), axis=-1)), -1)

        # positional encoding
        self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.enc)[1]), 0), [tf.shape(self.enc)[0], 1]),
                              vocab_size=self.option.max_len,
                              num_units=self.option.num_unit,
                              zero_pad=False,
                              scale=False,
                              scope="enc_pe")
        self.enc *= key_masks

        ## Dropout
        self.enc = tf.layers.dropout(self.enc,
                                     rate=0.1,
                                     training=tf.convert_to_tensor(self.option.is_training))

        ## Blocks
        for i in range(self.option.num_layer):
            with tf.variable_scope("num_blocks_{}".format(i)):
                ### Multihead Attention
                self.enc = multihead_attention(queries=self.enc,
                                               keys=self.enc,
                                               num_units=self.option.num_unit,
                                               num_heads=self.option.num_head,
                                               dropout_rate=0.1,
                                               is_training=self.option.is_training,
                                               causality=False)

                ### Feed Forward
                self.enc = feedforward(self.enc, num_units=[4 * self.option.num_unit, self.option.num_unit])
        return self.enc
