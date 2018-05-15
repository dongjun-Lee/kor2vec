# kor2vec
  Library for Korean morpheme and word vector representation.

- Paper : http://kiise.or.kr/e_journal/2018/5/JOK/pdf/04.pdf

## Requirements
For training,
- **Python 3**
- **Tensorflow**
- numpy, scipy
- Konlpy (Twitter)

For test and visualization,
- gensim
- sklearn
- matplotlib

## Model
<img width="458" alt="model" src="https://cloud.githubusercontent.com/assets/6512394/26795584/2a636ba4-4a61-11e7-96fd-2a79dcd2464c.png">
We define each word as a set of its morphemes, and a word vector is represented by the sum of the vector of its morphemes.

## Train Vectors
In order to learn morpheme vectors, do:

```
$ python3 train.py <input_corpus>
```
<input_corpus> format : one sentence = one line


### Change Hyperparameters

```
$ python3 train.py -h
usage: train.py [-h] [--embedding_size EMBEDDING_SIZE]
                [--window_size WINDOW_SIZE] [--min_count MIN_COUNT]
                [--num_sampled NUM_SAMPLED] [--learning_rate LEARNING_RATE]
                [--sampling_rate SAMPLING_RATE] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE]
                input

positional arguments:
  input                 input text file for training: one sentence per line

optional arguments:
  -h, --help            show this help message and exit
  --embedding_size EMBEDDING_SIZE
                        embedding vector size (default=150)
  --window_size WINDOW_SIZE
                        window size (default=5)
  --min_count MIN_COUNT
                        minimal number of word occurences (default=5)
  --num_sampled NUM_SAMPLED
                        number of negatives sampled (default=50)
  --learning_rate LEARNING_RATE
                        learning rate (default=1.0)
  --sampling_rate SAMPLING_RATE
                        rate for subsampling frequent words (default=0.0001)
  --epochs EPOCHS       number of epochs (default=3)
  --batch_size BATCH_SIZE
                        batch size (default=150)

```

### Load Trained Morpheme Vectors
```
$ python3
>>>> from gensim.models.keyedvectors import KeyedVectors
>>>> pos_vectors = KeyedVectors.load_word2vec_format('pos.vec', binary=False)
>>>> pos_vectors.most_similar("('대통령','Noun')")
```

### Generate Word Vectors
  A word vector is defined by sum of its morphemes' vectors.
```
$ python3
>>>> from konlpy.tag import Twitter
>>>> import numpy as np
>>>> twitter = Twitter()
>>>> word = "대통령이"
>>>> pos_list = twitter.pos(word, norm=True)
>>>> word_vector = np.sum([pos_vectors.word_vec(str(pos).replace(" ", "")) for pos in pos_list], axis=0)
```


## Test Dataset
- [Word Similarity Test](test_dataset/kor_ws353.csv) : Translated WordSim 353 Dataset into Korean. Translation ambiguous words were excluded.
  - WordSim 353 Dataset : http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.html
- Word Analogy Test : Created [Semantic Pair](test_dataset/kor_analogy_semantic.txt) 420 questions + [Syntactic Pair](test_dataset/kor_analogy_syntactic.txt) 840 questions.

## Test Morpheme Vectors
### Similarity Test
  Word similarity test using [kor_ws353.csv](test_dataset/kor_ws353.csv).
 ```
 $ python3 test/similarity_test.py pos.vec
 ```

### Analogy Test (Semantic)
  Word analogy test using [kor_analogy_semantic.txt](test_dataset/kor_analogy_semantic.txt).
 ```
 $ python3 test/analogy_test.py pos.vec
 ```
### Visualization
  Visualize the learned embeddings on two dimensional space using PCA.
```
$ python3 test/visualization.py pos.vec --words 밥 밥을 물 물을
```

## Donwload Pre-trained Morpheme Vectors
Morpheme vectors are trained on Naver news corpus (218M tokens) using our model. You can download pre-trained morpheme vectors here : http://mmlab.snu.ac.kr/~djlee/pos.vec

### Load Vectors using Gensim Library
```
$ python3
>>>> from gensim.models.keyedvectors import KeyedVectors
>>>> pos_vectors = KeyedVectors.load_word2vec_format('pos.vec', binary=False)
>>>> pos_vectors.most_similar("('대통령','Noun')")
>>>> pos_vectors.most_similar(positive=["('도쿄','Noun')", "('프랑스','Noun')"], negative=["('일본','Noun')"])
```



