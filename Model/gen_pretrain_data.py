import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import collections
import functools
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import six
import time
import math
from vocab import FreqVocab

tf.logging.set_verbosity(tf.logging.INFO)

random_seed = 12345
rng = random.Random(random_seed)

short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("pool_size", 10, "multiprocesses pool size.")
flags.DEFINE_integer("max_seq_length", 100, "max sequence length.")
flags.DEFINE_float("masked_lm_prob", 0.8, "Masked LM probability.")
flags.DEFINE_float("mask_prob", 1.0, "mask probabaility")
flags.DEFINE_bool("do_eval", False, "")
flags.DEFINE_bool("do_embed", True, "")
flags.DEFINE_integer("dupe_factor", 10, "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_string("data_dir", './inter_data/', "data dir.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_string("bizdate", None, "the signature of running experiments")

if FLAGS.bizdate is None:
    raise ValueError("bizdate is required.")

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(
    ",")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

MAX_PREDICTIONS_PER_SEQ = math.ceil(FLAGS.max_seq_length * FLAGS.masked_lm_prob)
SLIDING_STEP = round(FLAGS.max_seq_length * 0.6)

print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
print("MAX_PREDICTIONS_PER_SEQ:", MAX_PREDICTIONS_PER_SEQ)
print("SLIDING_STEP:", SLIDING_STEP)

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address, tokens, masked_lm_positions, masked_lm_labels):

        self.address = [address]
        self.tokens = list(map(lambda x: x[0], tokens))
        self.block_timestamps = list(map(lambda x: x[2], tokens))
        self.values = list(map(lambda x: x[3], tokens))

        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        self.io_flags = list(map(map_io_flag, tokens))
        self.cnts = list(map(lambda x: x[5], tokens))
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = "address: %s\n" % (self.address[0])
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    for inst_index in tqdm(range(len(instances))):
        instance = instances[inst_index]
        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        address = vocab.convert_tokens_to_ids(instance.address)
        counts = instance.cnts
        block_timestamps = instance.block_timestamps
        values = instance.values
        io_flags = instance.io_flags
        positions = convert_timestamp_to_position(block_timestamps)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["address"] = create_int_feature(address)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_positions"] = create_int_feature(positions)
        features["input_counts"] = create_int_feature(counts)
        features["input_io_flags"] = create_int_feature(io_flags)
        features["input_values"] = create_int_feature(values)

        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0


def main():
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open(FLAGS.data_dir + "eoa2seq_" + FLAGS.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename + "." + FLAGS.bizdate
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    print("===========Original===========")
    length_list = []
    for eoa in eoa2seq.keys():
        seq = eoa2seq[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq num:", len(length_list))

    # clip
    max_num_tokens = FLAGS.max_seq_length - 1
    seqs = []
    idx = 0
    for eoa, seq in eoa2seq.items():
        if len(seq) <= max_num_tokens:
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq
            idx += 1
        elif len(seq) > max_num_tokens:
            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * SLIDING_STEP))
            beg_idx.append(0)

            if len(beg_idx) > 500:
                beg_idx = list(np.random.permutation(beg_idx)[:500])
                for i in beg_idx:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

            else:
                for i in beg_idx[::-1]:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

    if FLAGS.do_embed:
        print("===========Generate Embedding Samples==========")
        write_instance = gen_embedding_samples(seqs)
        output_filename = FLAGS.data_dir + "embed.tfrecord" + "." + FLAGS.bizdate
        tf.logging.info("*** Writing to output embedding files ***")
        tf.logging.info("  %s", output_filename)

        write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                        MAX_PREDICTIONS_PER_SEQ, vocab,
                                        [output_filename])

    seqs = np.random.permutation(seqs)

    if FLAGS.do_eval:  # select 20% for testing
        print("========Generate Evaluation Samples========")
        eval_seqs = seqs[:round(len(seqs) * 0.2)]
        seqs = seqs[round(len(seqs) * 0.2):]

        eval_normal_instances = gen_samples(eval_seqs,
                                            dupe_factor=FLAGS.dupe_factor,
                                            masked_lm_prob=FLAGS.masked_lm_prob,
                                            max_predictions_per_seq=MAX_PREDICTIONS_PER_SEQ,
                                            pool_size=FLAGS.pool_size,
                                            rng=rng)

        eval_write_instance = eval_normal_instances
        rng.shuffle(eval_write_instance)
        eval_output_filename = FLAGS.data_dir + "test.tfrecord" + "." + FLAGS.bizdate
        tf.logging.info("*** Writing to Testing files ***")
        tf.logging.info("  %s", eval_output_filename)

        write_instance_to_example_files(eval_write_instance, FLAGS.max_seq_length,
                                        MAX_PREDICTIONS_PER_SEQ, vocab,
                                        [eval_output_filename])

    print("========Generate Training Samples========")
    normal_instances = gen_samples(seqs,
                                   dupe_factor=FLAGS.dupe_factor,
                                   masked_lm_prob=FLAGS.masked_lm_prob,
                                   max_predictions_per_seq=MAX_PREDICTIONS_PER_SEQ,
                                   pool_size=FLAGS.pool_size,
                                   rng=rng)

    write_instance = normal_instances
    rng.shuffle(write_instance)

    output_filename = FLAGS.data_dir + "train.tfrecord" + "." + FLAGS.bizdate
    tf.logging.info("*** Writing to Training files ***")
    tf.logging.info("  %s", output_filename)

    write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                    MAX_PREDICTIONS_PER_SEQ, vocab,
                                    [output_filename])

    return

if __name__ == '__main__':
    main()

