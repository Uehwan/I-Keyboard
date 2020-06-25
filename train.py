import logging
import logging.handlers
import tensorflow as tf
from data import Preprocessor
from model import BaseModel, AttnModel, HierarchModel, BP, TransModel
from options import TrainOptions
from utils import batch_wer, batch_cer


def expand(x):
    '''
    Hack. Because padded_batch doesn't play nice with scalars, so we expand the scalar to a vector of length 1
    :param x:
    :return:
    '''
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']), 0)
    return x


def deflate(x):
    '''
    Undo Hack. We undo the expansion we did in expand this is test
    '''
    x['length'] = tf.squeeze(x['length'], axis=1)
    return x


def data_augmentation(x):
    '''
    div_factor = tf.constant(2, dtype=tf.int64)
    min_length = tf.constant(3, dtype=tf.int64)

    idx_start = tf.random.uniform(
        shape=[], minval=0, maxval=tf.floordiv(tf.reduce_max(x['length']), div_factor), dtype=tf.int64
    )
    idx_length = tf.random.uniform(
        shape=[], minval=0, maxval=tf.reduce_max(x['length']) - idx_start, dtype=tf.int64
    )

    # don't know why below does not work
    def crop():
        x['x_seq'] = tf.slice(x['x_seq'], [idx_start], [idx_length])
        x['y_seq'] = tf.slice(x['y_seq'], [idx_start], [idx_length])
        x['id_seq'] = tf.slice(x['id_seq'], [idx_start], [idx_length])
        x['length'] = idx_length
        return x

    x = tf.cond(
        idx_length > min_length,
        crop,
        lambda: x
    )

    x['x_seq'] = tf.cond(
        idx_length > min_length,
        lambda: tf.slice(x['x_seq'], [idx_start], [idx_length]),
        lambda: x['x_seq']
    )

    x['y_seq'] = tf.cond(
        idx_length > min_length,
        lambda: tf.slice(x['y_seq'], [idx_start], [idx_length]),
        lambda: x['y_seq']
    )

    x['id_seq'] = tf.cond(
        idx_length > min_length,
        lambda: tf.slice(x['id_seq'], [idx_start], [idx_length]),
        lambda: x['id_seq']
    )

    x['length'] = tf.cond(
        idx_length > min_length,
        lambda: idx_length,
        lambda: x['length']
    )
    '''

    DISPLAY_X_MAX = tf.constant(1920, dtype=tf.int64)
    DISPLAY_Y_MAX = tf.constant(1080, dtype=tf.int64)
    DISPLAY_OFFSET = tf.constant(25, dtype=tf.int64)

    min_x = tf.minimum(tf.reduce_min(x['x_seq']) + DISPLAY_OFFSET, 0)
    max_x = tf.maximum(DISPLAY_X_MAX - tf.reduce_max(x['x_seq']) - DISPLAY_OFFSET, 0)
    min_y = tf.minimum(tf.reduce_min(x['y_seq']) + DISPLAY_OFFSET, 0)
    max_y = tf.maximum(DISPLAY_Y_MAX - tf.reduce_max(x['x_seq']) - DISPLAY_OFFSET, 0)

    x['x_seq'] = tf.cond(
        min_x < max_x,
        lambda: x['x_seq'] + tf.random.uniform(shape=[], minval=min_x, maxval=max_x, dtype=tf.int64),
        lambda: x['x_seq']
    )

    x['y_seq'] = tf.cond(
        min_y < max_y,
        lambda: x['y_seq'] + tf.random.uniform(shape=[], minval=min_y, maxval=max_y, dtype=tf.int64),
        lambda: x['y_seq']
    )

    return x


def make_dataset(path, augment=True, batch_size=64):
    '''
    Makes  a Tensorflow dataset that is shuffled, batched and parsed according to BibPreppy.
    You can chain all the lines here, I split them into separate calls so I could comment easily
    :param path: The path to a tf record file
    :param path: The size of our batch
    :return: a Dataset that shuffles and is padded
    '''
    # Read a tf record file. This makes a dataset of raw TFRecords
    dataset = tf.data.TFRecordDataset([path])
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset = dataset.map(Preprocessor.parse, num_parallel_calls=5)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000)
    # In order the pad the dataset, I had to use this hack to expand scalars to vectors.
    if augment:
        dataset = dataset.map(data_augmentation)
    dataset = dataset.map(expand)
    # Batch the dataset so that we get batch_size examples in each batch.
    # Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
    dataset = dataset.padded_batch(batch_size, padded_shapes = {
        "length": 1,  # Likewise for the length of the sequence
        "id_seq": tf.TensorShape([None]),  # but the seqeunce is variable length, we pass that information to TF
        "x_seq": tf.TensorShape([None]),   # but the seqeunce is variable length, we pass that information to TF
        "y_seq": tf.TensorShape([None])    # but the seqeunce is variable length, we pass that information to TF
    }, drop_remainder=True)
    # Finally, we need to undo that hack from the expand function
    dataset = dataset.map(deflate)
    return dataset


def prepare_dataset_iterators(train_ds_name, test_ds_name, batch_size=64, augment=True):
    # Make a dataset from the train data
    train_ds = make_dataset(train_ds_name, augment=augment, batch_size=batch_size)
    # make a dataset from the valdiation data
    test_ds = make_dataset(test_ds_name, batch_size=batch_size)
    # Define an abstract iterator
    # Make an iterator object that has the shape and type of our datasets
    iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                               train_ds.output_shapes)

    # This is an op that gets the next element from the iterator
    next_element = iterator.get_next()
    # These ops let us switch and reinitialize every time we finish an epoch
    train_init_op = iterator.make_initializer(train_ds)
    test_init_op = iterator.make_initializer(test_ds)

    return next_element, train_init_op, test_init_op


if __name__ == "__main__":
    import os

    opt = TrainOptions().parse()
    train_ds_name, test_ds_name = './data_train_aug.tfrecords', './data_val_aug.tfrecords'

    # Create target Directory if don't exist
    dir_name = 'log'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    else:
        print("Directory ", dir_name, " already exists")

    log = logging.getLogger(opt.name)
    log.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler('./log/' + opt.name + '.txt')
    streamHandler = logging.StreamHandler()

    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

    # Make datasets for train and validation
    with tf.Graph().as_default():
        checkpoint_dir = opt.name + "/checkpoint"
        if not os.path.exists(opt.name):
            os.mkdir(opt.name)
            os.mkdir(checkpoint_dir)

        gs = tf.train.get_or_create_global_step()
        next_element, train_init_op, test_init_op = prepare_dataset_iterators(
            train_ds_name,
            test_ds_name,
            batch_size=opt.batch_size,
            augment=True
        )
        train_writer = tf.summary.FileWriter(opt.name + "/logs/train")
        test_writer = tf.summary.FileWriter(opt.name + "/logs/test")
        if opt.model == 'seq2seq_attention':
            M = AttnModel(next_element, gs=gs, option=opt)
        elif opt.model == 'transformer':
            M = TransModel(next_element, gs=gs, option=opt)
        elif opt.model == 'hierarchical':
            M = HierarchModel(next_element, gs=gs, option=opt)
        else:
            M = BaseModel(next_element, gs=gs, option=opt)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            counter, min_val_loss = [], 10000
            lr = opt.lr
            old_test_loss, new_test_loss = 100, 100
            for epoch in range(opt.num_epoch + opt.num_epoch_decay):
                log.info("###########################################")
                log.info("Epoch {ep}".format(ep=epoch))

                train_losses = []
                for num_augmentation in range(2):
                    # Initialize the iterator to consume training data
                    sess.run(train_init_op)

                    # if new_test_loss > old_test_loss:  # We're not generalizing
                    if epoch > opt.num_epoch:
                        # half the learning rate
                        lr = opt.lr * (opt.num_epoch + opt.num_epoch_decay - epoch) / opt.num_epoch_decay

                    while True:
                        # As long as the iterator is not empty
                        try:
                            _, summary, gs, loss = sess.run([M.train, M.write_op, M.gs, M.loss], feed_dict={M.lr: lr})
                            train_losses.append(loss)
                            train_writer.add_summary(summary, gs)
                            train_writer.flush()
                            if gs % opt.print_freq == 0:
                                if opt.model == 'seq2seq_attention':
                                    tar, out, lengths, infer = sess.run([M.target, M.preds, M.lengths, M.inference],
                                                                        feed_dict={M.lr: lr})
                                else:
                                    tar, out, lengths = sess.run([M.target, M.preds, M.lengths],
                                                                 feed_dict={M.lr: lr})
                                log.info("{gs}************************************************".format(gs=gs))
                                ground_truth = BP.ids_to_string(tar[0], lengths[0])
                                hypothesis = BP.ids_to_string(out[0], lengths[0])
                                print(ground_truth)
                                if opt.model == 'seq2seq_attention':
                                    # print('Inference result:')
                                    print(BP.ids_to_string(infer[0], lengths[0]))
                                else:
                                    print(hypothesis)
                                total_error_cer, total_length_cer = batch_cer(tar, out, BP, lengths)
                                total_error_wer, total_length_wer = batch_wer(tar, out, BP, lengths)

                                log.info("CER: {cer}".format(cer=100 * total_error_cer / total_length_cer))
                                log.info("WER: {wer}".format(wer=100 * total_error_wer / total_length_wer))
                                log.info("learning rate: {lr}".format(lr=lr))

                        except tf.errors.OutOfRangeError:
                            # If the iterator is empty stop the while loop
                            if num_augmentation == 1:
                                train_loss = sum(train_losses) / len(train_losses)
                                log.info("Train Loss: {tl}".format(tl=train_loss))
                            break

                # Initialize the iterator to provide validation data
                sess.run(test_init_op)
                # We'll store the losses from each batch to get an average
                val_losses = []
                while True:
                    # As long as the iterator is not empty
                    try:
                        if opt.model == "seq2seq_attention":
                            tar, lengths, infer, loss, summary, gs, _ = sess.run(
                                [M.target, M.lengths, M.inference, M.loss, M.write_op, M.gs, M.increment_gs],
                                feed_dict={M.lr: lr}
                            )
                            # print("TEST RESULTS")
                            # print(BP.ids_to_string(tar[0], lengths[0]))
                            # print(BP.ids_to_string(infer[0], lengths[0]))

                        else:
                            loss, summary, gs, _ = sess.run([M.loss, M.write_op, M.gs, M.increment_gs],
                                                            feed_dict={M.lr: lr})
                        val_losses.append(loss)
                        test_writer.add_summary(summary, gs)
                        test_writer.flush()
                    except tf.errors.OutOfRangeError:
                        # Update the average loss for the epoch
                        # old_test_loss = new_test_loss
                        val_loss = sum(val_losses) / len(val_losses)
                        log.info("Validation Loss: {vl}".format(vl=val_loss))
                        break

                # if over-fitting occurs, stop training
                if min_val_loss > val_loss:
                    counter.append(False)
                    min_val_loss = val_loss
                    lr = lr * (2 - opt.lr_decay)
                    log.info("Min. loss updated---")
                    saver.save(sess, checkpoint_dir + '/model', gs)
                else:
                    lr = lr * opt.lr_decay
                    counter.append(True)

                if all(counter[-10:]):
                    log.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    log.info("Training completed")
                    log.info("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    break
