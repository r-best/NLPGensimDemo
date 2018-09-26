import os
import multiprocessing
from gensim import corpora, models

TRAINING_DIR = './training_small/initial_corpus'
ADAPT_DIR = './training_small/adapt'

## 
# Takes in a filename and returns its contents
# as an array of lines
##
def readFile(file):
    x = []
    with open(file, 'r') as fp:
        x = fp.readlines()
    return x


##
# Main method
##
def main():
    # Read in training files and format them for use in the model
    initial_corpus = readFile(TRAINING_DIR)
    initial_corpus = [x.split() for x in initial_corpus]
    # print(initial_corpus)

    # Instantiate model
    model = models.Word2Vec(min_count=1, iter=1)
    model.build_vocab(initial_corpus)
    model.train(initial_corpus, total_examples=model.corpus_count, epochs=2)
    # print("Initial Model:", model, '\n')
    print("Initial similarity between 'word1' and 'word2':", model.wv.n_similarity(['word1'], ['word2']))
    print("Initial similarity between 'word1' and 'word3':", model.wv.n_similarity(['word1'], ['word3']))
    print("Initial word most similar to 'word1':", model.wv.most_similar_to_given('word1', ['word2', 'word3']), '\n')

    # Read in data that will be used to adapt the model
    adaptation_corpus = readFile(ADAPT_DIR)
    adaptation_corpus = [x.split() for x in adaptation_corpus]
    # print(adaptation_corpus)

    # Add new data to model
    model.build_vocab(adaptation_corpus, update=True)
    model.train(adaptation_corpus, total_examples=model.corpus_count, epochs=3)
    # print("Adapted Model:", model, '\n')
    print("Adapted similarity between 'word1' and 'word2':", model.wv.n_similarity(['word1'], ['word2']))
    print("Adapted similarity between 'word1' and 'word3':", model.wv.n_similarity(['word1'], ['word3']))
    print("Adapted word most similar to 'word1':", model.wv.most_similar_to_given('word1', ['word2', 'word3']), '\n')


if __name__ == '__main__':
    print()
    main()
