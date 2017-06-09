import numpy as np
import tensorflow as tf
import collections
from konlpy.tag import Twitter
import argparse
import re
import math
import random

'''
    Step 1 : Parse Arguments.
'''
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input text file for training: one sentence per line")
parser.add_argument("--embedding_size", type=int, help="embedding vector size (default=150)", default=150)
parser.add_argument("--window_size", type=int, help="window size (default=5)", default=5)
parser.add_argument("--min_count", type=int, help="minimal number of word occurences (default=5)", default=5)
parser.add_argument("--num_sampled", type=int, help="number of negatives sampled (default=50)", default=50)
parser.add_argument("--learning_rate", type=float, help="learning rate (default=1.0)", default=1.0)
parser.add_argument("--sampling_rate", type=int, help="rate for subsampling frequent words (default=0.0001)", default=0.0001)
parser.add_argument("--epochs", type=int, help="number of epochs (default=3)", default=3)
parser.add_argument("--batch_size", type=int, help="batch size (default=150)", default=150)

args = parser.parse_args()

'''
    Step 2 : Pre-process Data.
'''

def build_dataset(train_text, min_count, sampling_rate):
    words = list()
    with open(train_text, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sentence = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', line).strip().split()
            if sentence:
                words.append(sentence)

    word_counter = [['UNK', -1]]
    word_counter.extend(collections.Counter([word for sentence in words for word in sentence]).most_common())
    word_counter = [item for item in word_counter if item[1] >= min_count or item[0] == 'UNK']

    word_dict = dict()
    for word, count in word_counter:
        word_dict[word] = len(word_dict)
    word_reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

    word_to_pos_li = dict()
    pos_list = list()
    twitter = Twitter()
    for w in word_dict:
        w_pos_li = list()
        for pos in twitter.pos(w, norm=True):
            w_pos_li.append(pos)

        word_to_pos_li[word_dict[w]] = w_pos_li
        pos_list += w_pos_li

    pos_counter = collections.Counter(pos_list).most_common()

    pos_dict = dict()
    for pos, _ in pos_counter:
        pos_dict[pos] = len(pos_dict)

    pos_reverse_dict = dict(zip(pos_dict.values(), pos_dict.keys()))

    word_to_pos_dict = dict()

    for word_id, pos_li in word_to_pos_li.items():
        pos_id_li = list()
        for pos in pos_li:
            pos_id_li.append(pos_dict[pos])
        word_to_pos_dict[word_id] = pos_id_li

    data = list()
    unk_count = 0
    for sentence in words:
        s = list()
        for word in sentence:
            if word in word_dict:
                index = word_dict[word]
            else:
                index = word_dict['UNK']
                unk_count += 1
            s.append(index)
        data.append(s)
    word_counter[0][1] = max(1, unk_count)

    data = sub_sampling(data, word_counter, word_dict, sampling_rate)

    return data, word_dict, word_reverse_dict, pos_dict, pos_reverse_dict, word_to_pos_dict


# Sub-sampling frequent words according to sampling_rate
def sub_sampling(data, word_counter, word_dict, sampling_rate):
    total_words = sum([len(sentence) for sentence in data])
    prob_dict = dict()
    for word, count in word_counter:
        f = count / total_words
        p = max(0, 1 - math.sqrt(sampling_rate / f))
        prob_dict[word_dict[word]] = p

    new_data = list()
    for sentence in data:
        s = list()
        for word in sentence:
            prob = prob_dict[word]
            if random.random() > prob:
                s.append(word)
        new_data.append(s)

    return new_data


data, word_dict, word_reverse_dict, pos_dict, pos_reverse_dict, word_to_pos_dict \
        = build_dataset(args.input, args.min_count, args.sampling_rate)

vocabulary_size = len(word_dict)
pos_size = len(pos_dict)
num_sentences = len(data)

print("number of sentences :", num_sentences)
print("vocabulary size :", vocabulary_size)
print("pos size :", pos_size)

pos_li = []
for key in sorted(pos_reverse_dict):
    pos_li.append(pos_reverse_dict[key])


'''
    Step 3 : Function to generate a training batch
'''

window_size = args.window_size
batch_size = args.batch_size


def generate_input_output_list(data, window_size):
    input_li = list()
    output_li = list()
    for sentence in data:
        for i in range(len(sentence)):
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    if sentence[i]!=word_dict['UNK'] and sentence[j]!=word_dict['UNK']:
                        input_li.append(sentence[i])
                        output_li.append(sentence[j])
    return input_li, output_li

input_li, output_li = generate_input_output_list(data, window_size)
input_li_size = len(input_li)


def generate_batch(iter, batch_size):
    index = (iter % (input_li_size//batch_size)) * batch_size
    batch_input = input_li[index:index+batch_size]
    batch_output_li = output_li[index:index+batch_size]
    batch_output = [[i] for i in batch_output_li]

    return np.array(batch_input), np.array(batch_output)


'''
    Step 4 : Build a model.
'''

embedding_size = args.embedding_size
num_sampled = args.num_sampled
learning_rate = args.learning_rate

valid_size = 20     # Random set of words to evaluate similarity on.
valid_window = 200  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    # Input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    words_matrix = [tf.placeholder(tf.int32, shape=None) for _ in range(batch_size)]
    vocabulary_matrix = [tf.placeholder(tf.int32, shape=(None)) for _ in range(vocabulary_size)]
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        pos_embeddings = tf.Variable(tf.random_uniform([pos_size, embedding_size], -1.0, 1.0), name='pos_embeddings')

        word_vec_list = []
        for i in range(batch_size):
            word_vec = tf.reduce_sum(tf.nn.embedding_lookup(pos_embeddings, words_matrix[i]), 0)
            word_vec_list.append(word_vec)
        word_embeddings = tf.stack(word_vec_list)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)), name='nce_weights'
        )
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='nce_biases')

    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=word_embeddings,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    # Compute the cosine similarity between minibatch exaples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(pos_embeddings), 1, keep_dims=True))
    normalized_embeddings = pos_embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


# Function to save vectors.
def save_model(pos_list, embeddings, file_name):
    with open(file_name, 'w') as f:
        f.write(str(len(pos_list)))
        f.write(" ")
        f.write(str(embedding_size))
        f.write("\n")
        for i in range(len(pos_list)):
            pos = pos_list[i]
            f.write(str(pos).replace("', '", "','") + " ")
            f.write(' '.join(map(str, embeddings[i])))
            f.write("\n")


'''
    Step 5 : Train a model.
'''

num_iterations = input_li_size // batch_size
print("number of iterations for each epoch :", num_iterations)
epochs = args.epochs
num_steps = num_iterations * epochs + 1

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized - Tensorflow")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(step, batch_size)

        word_list = []
        for word in batch_inputs:
            word_list.append(word_to_pos_dict[word])

        feed_dict = {}
        for i in range(batch_size):
            feed_dict[words_matrix[i]] = word_list[i]
        feed_dict[train_inputs] = batch_inputs
        feed_dict[train_labels] = batch_labels

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        if step % 20000 == 0:
            pos_embed = pos_embeddings.eval()

            # Print nearest words
            sim = similarity.eval()
            for i in range(valid_size):
                valid_pos = pos_reverse_dict[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % str(valid_pos)
                for k in range(top_k):
                    close_word = pos_reverse_dict[nearest[k]]
                    log_str = '%s %s,' % (log_str, str(close_word))
                print(log_str)

    # Save vectors
    save_model(pos_li, pos_embeddings.eval(), "pos.vec")












