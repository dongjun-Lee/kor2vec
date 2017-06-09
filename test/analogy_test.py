from gensim.models.keyedvectors import KeyedVectors
from konlpy.tag import Twitter
import argparse

testset = "test_dataset/kor_analogy_semantic.txt"
twitter = Twitter()


def analogy_test(testset, pos_vectors):
    correct, total, missed = 0, 0, 0
    with open(testset, 'r') as lines:
        for line in lines:
            if line.startswith("#") or len(line) <= 1:
                continue

            words = line.strip().split(" ")
            poss = list()
            for word in words:
                poss.append("('%s','Noun')" % word)
            total += 1

            try:
                similar_pos = pos_vectors.most_similar(positive=[poss[1], poss[2]], negative=[poss[0]])
                if similar_pos[0][0] == poss[3]:
                    correct += 1

            except KeyError:
                missed += 1

    return correct, total, missed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_file", type=str, help="trained morpheme vectors file")
    args = parser.parse_args()

    pos_vectors = KeyedVectors.load_word2vec_format(args.pos_file, binary=False)

    correct, total, missed = analogy_test(testset, pos_vectors)
    print("Missing :", str(missed) + "/" + str(total))
    print("Correct :", str(correct) + "/" + str(total) + " = " + str(correct/total))


