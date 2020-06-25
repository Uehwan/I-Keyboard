from collections import defaultdict, Counter
import numpy as np
import tensorflow as tf
import glob
import os
import random

PAD = "<PAD>"
START = "<START>"
EOS = "<EOS>"
ENTER = "<ENTER>"
DISPLAY_X_MAX = 1920
DISPLAY_Y_MAX = 1080
DISPLAY_OFFSET = 25


class Preprocessor:
    """
    Pre-process raw input files in the '.txt' form

    1) Read all the raw data files in PATH
    2) Tokenize each sequence and convert to numpy arrays
    3) Assigns ids sequentially to the token on the fly
    4) Store the resulting sequences as TFRecord files
    """
    def __init__(self, tokenizer_fn, path_to_files, result_file_name, data_aug=0, word_crop=False, append=False):
        self.vocab = defaultdict(self._next_value)  # map tokens to ids. Automatically gets next id when needed
        self.token_counter = Counter()  # Counts the token frequency
        self.vocab[ENTER] = 0
        self.next = 0  # After 3 comes 4
        self.tokenizer = tokenizer_fn
        self.path_to_files = path_to_files
        self.reverse_vocab = {}
        self.data_aug = data_aug
        self.result_file_name = result_file_name
        self.word_crop = word_crop
        self.append = append  # whether to append start and eos to each sentence

    def _next_value(self):
        self.next += 1
        return self.next

    def prepare_data(self):
        # list up the raw data files
        full_path = os.path.join(self.path_to_files, '*.txt')
        file_list = glob.glob(full_path)

        # randomly shuffle files and select one for test
        random.shuffle(file_list)
        train_file_list, val_file_list, test_file = file_list[:-3], file_list[-3:-1], file_list[-1:]
        train_filename, val_filename, test_filename = self.result_file_name

        self._write_to_tfrecord(train_file_list, train_filename, self.data_aug)
        self._write_to_tfrecord(val_file_list, val_filename, 0)  # no need to augment data for val & test
        self._write_to_tfrecord(test_file, test_filename, 0)  # no need to augment data for val & test

    def _write_to_tfrecord(self, file_list, target_name, data_aug):
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(target_name)
        for train_file in file_list:
            print("Processing: ", train_file, "...")
            char, word, x_seq, y_seq = self._process_one_file(train_file)
            if data_aug and not self.word_crop:
                char, word, x_seq, y_seq = self._data_augmentation(char, word, x_seq, y_seq, data_aug)
            for c, w, x, y in zip(char, word, x_seq, y_seq):
                example = self._sequence_to_tf_example(c, w, x, y)
                writer.write(example.SerializeToString())
        writer.close()

    @staticmethod
    def _data_augmentation(char, word, x_seq, y_seq, data_aug):
        aug_char, aug_word, aug_x_seq, aug_y_seq = [], [], [], []
        for c, w, x, y in zip(char, word, x_seq, y_seq):
            _, _, _, _ = aug_char.append(c), aug_word.append(w), aug_x_seq.append(x), aug_y_seq.append(y)
            min_x, max_x = min(-min(x) + DISPLAY_OFFSET, 0), max(DISPLAY_X_MAX - max(x) - DISPLAY_OFFSET, 0)
            min_y, max_y = min(-min(y) + DISPLAY_OFFSET, 0), max(DISPLAY_Y_MAX - max(y) - DISPLAY_OFFSET, 0)
            for _ in range(data_aug):
                res_x, res_y = random.randint(min_x, max_x), random.randint(min_y, max_y)
                _, _ = aug_char.append(c), aug_word.append(w)
                new_x, new_y = [e + res_x for e in x], [e + res_y for e in y]
                _, _ = aug_x_seq.append(new_x), aug_y_seq.append(new_y)

        return aug_char, aug_word, aug_x_seq, aug_y_seq

    @staticmethod
    def _append_results(cc, ww, xx, yy, char, word, x_seq, y_seq):
        if type(cc[0]) is str:
            char.append(cc), word.append(ww), x_seq.append(xx), y_seq.append(yy)
        else:
            _, _ = [char.append(ce) for ce in cc], [word.append(we) for we in ww]
            _, _ = [x_seq.append(xe) for xe in xx], [y_seq.append(ye) for ye in yy]

        return char, word, x_seq, y_seq

    def _split_sentence(self, line, x_seq_line, y_seq_line, length_of_first=0, enter=None):
        char = self.tokenizer(line.replace('\n', ''))
        word = line.replace('\n', '').split()
        x_seq = list(map(int, x_seq_line.replace('\n', '').split(',')))
        y_seq = list(map(int, y_seq_line.replace('\n', '').split(',')))

        if enter:
            char, word = char + [ENTER], word + [ENTER]
            x_seq, y_seq = x_seq[:length_of_first], y_seq[:length_of_first]
        elif length_of_first:
            x_seq, y_seq = x_seq[length_of_first:], y_seq[length_of_first:]

        if random.random() < 0.5:
            if enter:
                char, x_seq, y_seq = char[:-2] + char[-1:], x_seq[:-2] + x_seq[-1:], y_seq[:-2] + y_seq[-1:]
            else:
                char, x_seq, y_seq = char[:-1], x_seq[:-1], y_seq[:-1]

        split_criteria = len(word)
        if split_criteria < 4 or not self.word_crop or random.random() < 0.3:
            return char, word, x_seq, y_seq

        if split_criteria < 11:
            num_of_chunks = 2 if random.random() < 0.5 else 3
        else:
            num_of_chunks = 3 if random.random() < 0.5 else 4

        num_per_chunks = split_criteria // num_of_chunks
        num_remain_chunks = split_criteria % num_of_chunks
        result_chars, result_words, result_x_seqs, result_y_seqs = [], [], [], []

        num_to_iter = num_of_chunks - 1 if not num_remain_chunks else num_of_chunks
        for _ in range(num_to_iter):
            num_of_space = num_per_chunks - 1
            len_of_words = sum([len(w) for w in word[:num_per_chunks]])
            where_to_chunk = len_of_words + num_of_space
            if len(char[:where_to_chunk]) > 2:
                _, _ = result_chars.append(char[:where_to_chunk]), result_words.append(word[:num_per_chunks])
                _, _ = result_x_seqs.append(x_seq[:where_to_chunk]), result_y_seqs.append(y_seq[:where_to_chunk])
            char, word = char[where_to_chunk + 1:], word[num_per_chunks:]
            x_seq, y_seq = x_seq[where_to_chunk + 1:], y_seq[where_to_chunk + 1:]

        # last chunk => remove period probabilistically
        if not word[0] == ENTER:
            if len(char) > 2:
                _, _ = result_chars.append(char), result_words.append(word)
                _, _ = result_x_seqs.append(x_seq), result_y_seqs.append(y_seq)

        return result_chars, result_words, result_x_seqs, result_y_seqs

    def _process_one_file(self, file_name):
        # readout the file and turn it into a list
        with open(file_name, "r") as f:
            lines = f.readlines()

        char, word, x_seq, y_seq = [], [], [], []

        line_iterator = iter(lines)
        for line in line_iterator:
            next_line = next(line_iterator)
            number_included = self._check_if_contains_number(line)
            coord = self._check_if_coordinate_line(next_line)
            if coord:  # the next line contains y-coordinates
                second_next_line = next(line_iterator)
                if not number_included:
                    #if self.word_crop:
                    for _ in range(self.data_aug):
                        cc, ww, xx, yy = self._split_sentence(line, next_line, second_next_line)
                        char, word, x_seq, y_seq = self._append_results(cc, ww, xx, yy, char, word, x_seq, y_seq)
                    '''
                    else:
                        # append sentence
                        char.append(self.tokenizer(line.replace('\n', '')))
                        word.append(line.replace('\n', '').split())
                        # append coordinates
                        x_seq.append(list(map(int, next_line.replace('\n', '').split(','))))
                        y_seq.append(list(map(int, second_next_line.replace('\n', '').split(','))))
                    '''

            else:  # two sentences connected by enter
                second_line_number_included = self._check_if_contains_number(next_line)
                x_seq_line, y_seq_line = next(line_iterator), next(line_iterator)
                length_of_first = len(line)
                if not number_included:
                    # if self.word_crop:
                    for _ in range(self.data_aug):
                        cc, ww, xx, yy = self._split_sentence(line, x_seq_line, y_seq_line, length_of_first, ENTER)
                        char, word, x_seq, y_seq = self._append_results(cc, ww, xx, yy, char, word, x_seq, y_seq)
                    '''
                    else:
                        char.append(self.tokenizer(line.replace('\n', '')) + [ENTER])
                        word.append(line.replace('\n', '').split() + [ENTER])
                        x_seq.append(list(map(int, x_seq_line.replace('\n', '').split(',')[:length_of_first])))
                        y_seq.append(list(map(int, y_seq_line.replace('\n', '').split(',')[:length_of_first])))
                    '''
                if not second_line_number_included:
                    # if self.word_crop:
                    for _ in range(self.data_aug):
                        cc, ww, xx, yy = self._split_sentence(next_line, x_seq_line, y_seq_line, length_of_first)
                        char, word, x_seq, y_seq = self._append_results(cc, ww, xx, yy, char, word, x_seq, y_seq)
                    '''
                    else:
                        char.append(self.tokenizer(next_line.replace('\n', '')))
                        word.append(next_line.replace('\n', '').split())
                        x_seq.append(list(map(int, x_seq_line.replace('\n', '').split(',')[length_of_first:])))
                        y_seq.append(list(map(int, y_seq_line.replace('\n', '').split(',')[length_of_first:])))
                    '''
        return char, word, x_seq, y_seq

    @staticmethod
    def _check_if_coordinate_line(one_line):
        return all(char.isdigit() for char in one_line.replace('\n', '').split(','))

    @staticmethod
    def _check_if_contains_number(one_line):
        return any(char.isdigit() for char in list(one_line.replace('\n', '')))

    def _sequence_to_tf_example(self, char, word, x_seq, y_seq):
        """
        Gets a sequence (a text like "hello how are you") and returns a a SequenceExample
        :param sequence: Some text
        :return: A A sequence exmaple
        """
        # Convert the text to a list of ids
        id_list = self.sentence_to_id_list(char)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        if self.append:
            sequence_length = len(id_list) + 2  # For start and end
        else:
            sequence_length = len(id_list)

        # Add the context feature, here we just need length
        ex.context.feature["length"].int64_list.value.append(sequence_length)
        # Feature lists for the two sequential features of our example
        # Add the tokens. This is the core sequence.
        # You can add another sequence in the feature_list dictionary, for translation for instance
        fl_tokens = ex.feature_lists.feature_list["tokens"]
        x_sequence = ex.feature_lists.feature_list["x_sequence"]
        y_sequence = ex.feature_lists.feature_list["y_sequence"]

        if self.append:
            # Prepend with start token
            fl_tokens.feature.add().int64_list.value.append(self.vocab[START])
            x_sequence.feature.add().int64_list.value.append(self.vocab[START])
            y_sequence.feature.add().int64_list.value.append(self.vocab[START])

        for token, x, y in zip(id_list, x_seq, y_seq):
            # Add those tokens one by one
            fl_tokens.feature.add().int64_list.value.append(token)
            x_sequence.feature.add().int64_list.value.append(x)
            y_sequence.feature.add().int64_list.value.append(y)

        if self.append:
            # append  with end token
            fl_tokens.feature.add().int64_list.value.append(self.vocab[EOS])
            x_sequence.feature.add().int64_list.value.append(self.vocab[EOS])
            y_sequence.feature.add().int64_list.value.append(self.vocab[EOS])
        return ex

    def ids_to_string(self, tokens, length=None):
        string = ''.join([self.reverse_vocab[x] for x in tokens[:length]])
        return string

    def _convert_token_to_id(self, token):
        """
        Gets a token, looks it up in the vocabulary. If it doesn't exist in the vocab,
        it gets added to id with an id. Then, we return the id
        :param token:
        :return: the token id in the vocab
        """
        self.token_counter[token] += 1
        return self.vocab[token]

    def _tokens_to_id_list(self, tokens):
        return list(map(self._convert_token_to_id, tokens))

    def sentence_to_id_list(self, sent):
        id_list = self._tokens_to_id_list(sent)
        return id_list

    def sentence_to_numpy_array(self, sent):
        id_list = self.sentence_to_id_list(sent)
        return np.array(id_list)

    def update_reverse_vocab(self):
        self.reverse_vocab = {id_: token for token, id_ in self.vocab.items()}

    def id_list_to_text(self, id_list):
        tokens = ''.join(map(lambda x: self.reverse_vocab[x], id_list))
        return tokens

    @staticmethod
    def parse(ex):
        """
        Explain to TF how to go from a serialized example back to tensors
        :param ex:
        :return: A dictionary of tensors, in this case {seq: The sequence, length: The length of the sequence}
        """
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "x_sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "y_sequence": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"id_seq": sequence_parsed["tokens"],
                "x_seq": sequence_parsed["x_sequence"],
                "y_seq": sequence_parsed["y_sequence"],
                "length": context_parsed["length"]}


if __name__ == "__main__":
    import pickle

    # how many times to augment the data
    # total num: original_size * (1 + DATA_AUGMENTATION)
    result_file_name = ['./data_train_aug.tfrecords', './data_val_aug.tfrecords', './data_test_aug.tfrecords']
    result_obj_name = './data_processor_aug.pkl'

    data_processor = Preprocessor(list, "./raw_data/", result_file_name, word_crop=True, data_aug=4)
    data_processor.prepare_data()
    data_processor.update_reverse_vocab()
    pickle.dump(data_processor, open(result_obj_name, 'wb'))
