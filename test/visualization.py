from gensim.models.keyedvectors import KeyedVectors
from konlpy.tag import Twitter
import numpy as np
import numpy.linalg as la
import argparse
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sys

twitter = Twitter()


def normalize(array):
    norm = la.norm(array)
    return array / norm


def create_word_vector(word, pos_embeddings):
    pos_list = twitter.pos(word, norm=True)
    word_vector = np.sum([pos_vectors.word_vec(str(pos).replace(" ", "")) for pos in pos_list], axis=0)
    return normalize(word_vector)


def plot_with_labels(embeds, labels, filename="output.png"):
    plt.figure(figsize=(18, 18))
    pca = decomposition.PCA(n_components=2)
    pca.fit(embeds)
    Y = pca.transform(embeds)
    for i, label in enumerate(labels):
        x, y = Y[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
        plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_file", type=str, help="trained morpheme vectors file")
    parser.add_argument('-w', '--words', type=str, nargs='+')
    args = parser.parse_args()

    if not args.words:
        print("Input words are empty.")
        sys.exit()

    pos_vectors = KeyedVectors.load_word2vec_format(args.pos_file, binary=False)
    words = args.words
    word_embeddings = list()

    for word in words:
        word_embed = create_word_vector(word, pos_vectors)
        word_embeddings.append(word_embed)

    plot_with_labels(np.array(word_embeddings), words)
