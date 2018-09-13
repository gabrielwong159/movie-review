# Sentiment analysis on movie reviews
## About
<img align="right" src="https://imgur.com/sz1Xguk.png" />

This project is an exploration of methods involved in natural language processing, and in this case, sentiment analysis with text classification.

The data is obtained from a Kaggle tutorial competition, [`Bag of Words Meets Bags of Popcorn`](https://www.kaggle.com/c/word2vec-nlp-tutorial/data), and consists of movie reviews from IMDB. The objective is to give a binary classification, indicating a positive or negative sentiment.

## Methodology
### Data preprocessing
Typical data cleaning steps for text include removing stop words and normalizing text. In this example, we also implement removal of HTML tags.

### Feature extraction
#### Bag-of-words word counts
In a bag-of-words model, the sequence of words within a sentence does not matter. Taking an example from [Bag of Words Meets Bags of Popcorn](#bag-of-words-meets-bags-of-popcorn), we have two sentences:

```
Sentence 1: "The cat sat on the hat"
Sentence 2: "The dog ate the cat and the hat"
```

Given that the vocabulary from the two sentences is `{the, cat, sat, on, hat, dog, ate, and}`, we simply construct a vector based on the number of occurrences of each word in a sentence.

| Sentence                        | the | cat | sat | on | hat | dog | ate | and |
|---------------------------------|-----|-----|-----|----|-----|-----|-----|-----|
| The cat sat on the hat          | 2   | 1   | 1   | 1  | 1   | 0   | 0   | 0   |
| The dog ate the cat and the hat | 3   | 1   | 0   | 0  | 1   | 1   | 1   | 1   |

Thus, the vectors produced for the sentences are:

```
Sentence 1: [2, 1, 1, 1, 1, 0, 0, 0]
Sentence 2: [3, 1, 0, 0, 1, 1, 1, 1]
```

Each sentence can thus be transformed into a vector, with the length of the vector being the number of words in the vocabulary. During classification, the model will then possibly learn that higher occurrences of certain words are more likely to lead to a particular prediction.

#### `word2vec`
`word2vec` produces word embeddings for one-hot encoded vectors, such that each word can be represented as an `n`-dimensional vector. These word vectors are able to capture word relations in vector space.

##### Continuous bag-of-words (CBOW)
In the CBOW model, we attempt to predict a **word** given its **context**. This means that we attempt to predict the center word from the sum of surrounding word vectors.

##### Skip-gram
For the skip-gram model, we predict the **context** given a **word**. Instead of predicting the center word from surounding words, given the center word, we instead predict each surrounding single word.

**The skip-gram model is found to perform better against words that appear less frequently.** However, it is much slower to train, as compared to the CBOW model.

Thus, **negative sampling** is introduced when training the skip-gram model. On top of the surrounding words, we also take `k` negative samples (words that are not within the context window).
During training, we maximimize the probability that the words in the context window appears, while also minimizing the probability that other words appear.

### Classification
#### Machine learning
We use a **random forest classifier** with `SciKit-Learn` as a ML classifier for extracted features.

#### RNN
The RNN model architecture consists of the following components:
1. Word embedding as a randomly initialized _trainable variable_
1. RNN for feature extraction
1. Fully-connected layer for softmax classification

#### CNN
Model architecture for CNN is obtained from the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

Similar to the RNN model architecture, the CNN involves training new word embeddings from scratch. After obtaining word vectors, the vector is independently passed through convolutional layers with kernels of sizes 3, 4, and 5. The max of each filter is then taken through max-pooling, resulting in a tensor of shape `[batch, 1, 1, n_filters]` per kernel size.

The results of each convolutional layer is then concatenated and flattened, before being fed into a fully-connected layer for softmax classification.


## Resources
### [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words)
This tutorial guides the user through constructing a **bag-of-words** classification model. It begins with data cleaning methods, before feature extraction by **word counts** and **word2vec**. Classification is done by **random forest**.

### [Predicting Movie Review Sentiment with TensorFlow and TensorBoard](https://medium.com/@Currie32/predicting-movie-review-sentiment-with-tensorflow-and-tensorboard-53bf16af0acf)
Guide to using RNNs for text classification.

### [Implementing a CNN for Text Classification in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

## Additional readings
### `word2vec`
* [`word2vec` lecture from Stanford CS224n](http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf)
* [Tensorflow `word2vec` tutorial](https://www.tensorflow.org/tutorials/representation/word2vechttps://www.tensorflow.org/tutorials/representation/word2vec)
* [Explanation of `word2vec` training](http://www.1-4-5.net/~dmm/ml/how_does_word2vec_work.pdf)
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)