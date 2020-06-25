import tensorflow as tf
import time
import logging
import logging.handlers
import os
from data import Preprocessor
from model import BaseModel, AttnModel, BP, TransModel, HierarchModel
from options import TrainOptions
from train import prepare_dataset_iterators
from utils import batch_wer, batch_cer


opt = TrainOptions().parse()
checkpoint_dir = opt.name + "/checkpoint"

train_ds_name, test_ds_name = './data_train_aug.tfrecords', './data_test_aug.tfrecords'

# Create target Directory if don't exist
dir_name = 'test_results'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("Directory ", dir_name, " Created ")
else:
    print("Directory ", dir_name, " already exists")

log = logging.getLogger(opt.name)
log.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('./test_results/' + opt.name + '.txt')
streamHandler = logging.StreamHandler()

log.addHandler(fileHandler)
log.addHandler(streamHandler)

with tf.Graph().as_default():
    tf.random.set_random_seed(1234)
    gs = tf.train.get_or_create_global_step()
    next_element, train_init_op, test_init_op = prepare_dataset_iterators(
        train_ds_name,
        test_ds_name,
        batch_size=opt.batch_size
    )

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
                    log.info("Ground-truth: {}".format(ground_truth))
                    log.info("Decode-result: {}".format(hypothesis))

            except tf.errors.OutOfRangeError:
                # If the iterator is empty stop the while loop
                log.info("Total Time spent: {}".format(time.time() - start_time))
                log.info("Total Num. of Char.s: {}".format(total_length_cer))
                log.info("Total Num. of Words: {}".format(total_length_wer))
                log.info("Total Result CER: {}".format(100 * total_error_cer / total_length_cer))
                log.info("Total Result WER: {}".format(100 * total_error_wer / total_length_wer))
                break
