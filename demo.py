import os
import multiprocessing
from gensim import corpora, models
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from matplotlib import pyplot

TRAINING_DIR = './training_small/initial_corpus'
ADAPT_DIR = './training_small/adapt'

## 
# Takes in a variadic number of filepaths and
# returns an array of all of their contents
# read as strings; if a directory is included
# all of its files will be recursively processed
##
def readFiles(file):
    x = []
    _readFiles(file, x)
    return x
##
# Recursive helper function for readFiles()
##
def _readFiles(file, text):
    if os.path.isdir(file):
        for item in os.listdir(file):
            _readFiles(os.path.join(file, item), text)
    else:
        with open(file, 'r') as fp:
            text.append("".join(fp.readlines()))


##
# Takes in a string, applies a set of 
# preprocessing rules to it, and splits
# it into a list of its tokens
##
def preprocess(text):
    text = text.replace('\n', ' \n ')

    tokens = text.split(' ')
    tokens = [x for x in tokens if x is not ""]

    # frequency = {}
    # for token in tokens:
    #     if token not in frequency:
    #         frequency[token] = 1
    #     else:
    #         frequency[token] += 1
    # tokens = [x for x in tokens if frequency[x] > 1]
    
    return tokens


##
# Main method
##
def main():
    # Read in training files and format them for use in the model
    initial_corpus = readFiles(TRAINING_DIR)
    initial_corpus = [preprocess(x) for x in initial_corpus]
    adaptation_corpus = readFiles(ADAPT_DIR)
    adaptation_corpus = [preprocess(x) for x in adaptation_corpus]

    # Instantiate model
    model = models.Word2Vec(initial_corpus, size=len(initial_corpus[0]), window=4, min_count=1, workers=multiprocessing.cpu_count(), alpha=0.025)
    print(model)
    print(model.wv['word1'])
    # Printed weights of word1:
    # [-0.01076334  0.00622735 -0.0402417   0.02217639  0.04611698  0.01181554  0.00560959  0.03880095 -0.04292683 -0.02537567]

    # Add new data to model
    model.train(adaptation_corpus, total_examples=model.corpus_count, epochs=model.iter)
    print(model)
    print(model.wv['word1'])
    # Prints different from before (weights have changed):
    # [-0.01776467  0.0080032  -0.03565837  0.02399305  0.05069579  0.00730562 0.00882716  0.03723516 -0.03821499 -0.03624377]


if __name__ == '__main__':
    main()
