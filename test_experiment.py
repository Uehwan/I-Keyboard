import tensorflow as tf
import time
import logging
import logging.handlers
import os
import glob
import numpy as np
from data import Preprocessor
from model import BaseModel, AttnModel, BP, TransModel, HierarchModel
from options import TrainOptions
from utils import batch_wer, batch_cer
from train import expand, deflate
from symspellpy.symspellpy import SymSpell, Verbosity


def parse_function(lengths, id_seq, x_seq, y_seq):
    return {'length': lengths, 'id_seq': id_seq, 'x_seq': x_seq, 'y_seq': y_seq}


opt = TrainOptions().parse()
checkpoint_dir = opt.name + "/checkpoint"

# Create target Directory if don't exist
dir_name = 'new_test_experiment_results'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("Directory ", dir_name, " Created ")
else:
    print("Directory ", dir_name, " already exists")

log = logging.getLogger(opt.name)
log.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('./new_test_experiment_results/' + opt.name + '.txt')
streamHandler = logging.StreamHandler()

log.addHandler(fileHandler)
log.addHandler(streamHandler)

PATH = '../190225_data_including_time'

# list up the raw data files
full_path = os.path.join(PATH, '*.txt')
file_list = glob.glob(full_path)


# maximum edit distance per dictionary precalculation
max_edit_distance_dictionary = 2
prefix_length = 5
# create object
# sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
# load dictionary
dictionary_path = os.path.join("symspellpy",
                               "frequency_dictionary_en_82_765.txt")
term_index = 0  # column of the term in the dictionary text file
count_index = 1  # column of the term frequency in the dictionary text file
# if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
#     print("Dictionary file not found")


def correct_typos(batch_in, lengths, sym_spell_obj, translator):
    max_edit_distance_lookup = 2
    for i in range(len(batch_in)):
        temp_sent = translator.ids_to_string(batch_in[i], lengths[i])
        corrected = sym_spell_obj.lookup_compound(temp_sent, max_edit_distance_lookup)[0].term
        corrected = translator.sentence_to_id_list(corrected)
        batch_in[i, :len(corrected)] = corrected
    return batch_in


for file in file_list:
    with open(file) as f:
        lines = f.readlines()
    log.info('analysis for temp')
    log.info('{}'.format(file))
    id_seq, x_seq, y_seq, lengths = [], [], [], []
    line_iterator = iter(lines)
    for line in line_iterator:
        x_seq_line, y_seq_line, _ = next(line_iterator), next(line_iterator), next(line_iterator)
        # x_seq_line, y_seq_line = next(line_iterator), next(line_iterator)
        # id_seq.append([BP.vocab['<START>']] + BP.sentence_to_id_list(list(line[:-1])) + [BP.vocab['<EOS>']])
        # x_seq.append([BP.vocab['<START>']] + x_seq_line.split(',') + [BP.vocab['<EOS>']])
        # y_seq.append([BP.vocab['<START>']] + y_seq_line.split(',') + [BP.vocab['<EOS>']])
        id_seq.append(BP.sentence_to_id_list(list(line[:-1])))
        x_seq.append(x_seq_line.split(','))
        y_seq.append(y_seq_line.split(','))
        lengths.append(len(id_seq[-1]))

    id_seq, x_seq, y_seq, lengths = np.array(id_seq), np.array(x_seq), np.array(y_seq), np.array(lengths)


    def gen():
        for i in range(len(id_seq)):
            yield (lengths[i], id_seq[i], x_seq[i], y_seq[i])

    with tf.Graph().as_default():
        tf.random.set_random_seed(1234)
        gs = tf.train.get_or_create_global_step()

        dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int64, tf.int64, tf.int64, tf.int64),
            (tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]))
        )
        # dataset = tf.data.Dataset.from_tensor_slices((id_seq, x_seq, y_seq, lengths))
        dataset = dataset.map(parse_function, num_parallel_calls=len(id_seq))
        dataset = dataset.map(expand)
        dataset = dataset.padded_batch(opt.batch_size, padded_shapes={
            "length": 1,  # Likewise for the length of the sequence
            "id_seq": tf.TensorShape([None]),  # but the seqeunce is variable length, we pass that information to TF
            "x_seq": tf.TensorShape([None]),  # but the seqeunce is variable length, we pass that information to TF
            "y_seq": tf.TensorShape([None])  # but the seqeunce is variable length, we pass that information to TF
        })
        dataset = dataset.map(deflate)

        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        next_element = iterator.get_next()
        test_init_op = iterator.make_initializer(dataset)

        if opt.model == 'seq2seq_attention':
            M = AttnModel(next_element, gs=gs, option=opt)
        elif opt.model == 'transformer':
            M = TransModel(next_element, gs=gs, option=opt)
        elif opt.model == 'hierarchical':
            M = HierarchModel(next_element, gs=gs, option=opt)
        else:
            M = BaseModel(next_element, gs=gs, option=opt)

        print("STEP 1. Model Construction Completed")

        with tf.Session() as sess:
            saver = tf.train.Saver()
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, latest_checkpoint)
            print("STEP 2. Model Restoration Completed")
            lr = 0.001
            sess.run(test_init_op)
            total_error_cer, total_length_cer = 0, 0
            total_error_wer, total_length_wer = 0, 0
            start_time = time.time()
            while True:
                try:
                    if opt.model == 'seq2seq_attention':
                        tar, out, lengths, infer, gs = sess.run([M.target, M.preds, M.lengths, M.inference, M.gs],
                                                                feed_dict={M.lr: lr})
                    else:
                        tar, out, lengths, gs = sess.run([M.target, M.preds, M.lengths, M.gs],
                                                         feed_dict={M.lr: lr})

                    if opt.model == 'seq2seq_attention':
                        # print('Inference result:')
                        batch_err_cer, batch_len_cer = batch_cer(tar, infer, BP, lengths)
                        total_error_cer = total_error_cer + batch_err_cer
                        total_length_cer = total_length_cer + batch_len_cer

                        batch_err_wer, batch_len_wer = batch_wer(tar, infer, BP, lengths)
                        total_error_wer = total_error_wer + batch_err_wer
                        total_length_wer = total_length_wer + batch_len_wer
                    else:
                        # out = correct_typos(out, lengths, sym_spell, BP)
                        batch_err_cer, batch_len_cer = batch_cer(tar, out, BP, lengths)
                        total_error_cer = total_error_cer + batch_err_cer
                        total_length_cer = total_length_cer + batch_len_cer

                        batch_err_wer, batch_len_wer = batch_wer(tar, out, BP, lengths)
                        total_error_wer = total_error_wer + batch_err_wer
                        total_length_wer = total_length_wer + batch_len_wer

                    if gs % opt.print_freq == 0:
                        ground_truth = BP.ids_to_string(tar[0], lengths[0])
                        if opt.model == 'seq2seq_attention':
                            hypothesis = BP.ids_to_string(infer[0], lengths[0])
                        else:
                            hypothesis = BP.ids_to_string(out[0], lengths[0])
                        log.info("{}-th Example decoding results".format(gs))
                        log.info("Ground: {}".format(ground_truth))
                        log.info("Decode: {}".format(hypothesis))

                except tf.errors.OutOfRangeError:
                    # If the iterator is empty stop the while loop
                    log.info("Total Time spent: {}".format(time.time() - start_time))
                    log.info("Total Num. of Char.s: {}".format(total_length_cer))
                    log.info("Total Num. of Words: {}".format(total_length_wer))
                    log.info("Total Result CER: {}".format(100 * total_error_cer / total_length_cer))
                    log.info("Total Result WER: {}".format(100 * total_error_wer / total_length_wer))
                    break
