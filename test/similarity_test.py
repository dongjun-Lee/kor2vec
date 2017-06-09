from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import numpy.linalg as la
from konlpy.tag import Twitter
import scipy.stats as st
import argparse

testset = "test_dataset/kor_ws353.csv"
twitter = Twitter()


def normalize(array):
    norm = la.norm(array)
    return array / norm


def create_word_vector(word, pos_vectors):
    pos_list = twitter.pos(word, norm=True)
    word_vector = np.sum([pos_vectors.word_vec(str(pos).replace(" ", "")) for pos in pos_list], axis=0)
    return normalize(word_vector)


def word_sim_test(filename, pos_vectors):
    delim = ','
    actual_sim_list, pred_sim_list = [], []
    missed = 0

    with open(filename, 'r') as pairs:
        for pair in pairs:
            w1, w2, actual_sim = pair.strip().split(delim)

            try:
                w1_vec = create_word_vector(w1, pos_vectors)
                w2_vec = create_word_vector(w2, pos_vectors)
                pred = float(np.inner(w1_vec, w2_vec))
                actual_sim_list.append(float(actual_sim))
                pred_sim_list.append(pred)

            except KeyError:
                missed += 1

    spearman, _ = st.spearmanr(actual_sim_list, pred_sim_list)
    pearson, _ = st.pearsonr(actual_sim_list, pred_sim_list)

    return spearman, pearson, missed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_file", type=str, help="trained morpheme vectors file")
    args = parser.parse_args()

    pos_vectors = KeyedVectors.load_word2vec_format(args.pos_file, binary=False)

    spearman, pearson, missed = word_sim_test(testset, pos_vectors)
    print("Missing words :", missed)
    print("Spearman coefficient :", spearman)
    print("Pearson coefficient :", pearson)
