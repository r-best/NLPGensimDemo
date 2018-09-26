import os
from gensim import corpora, models
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from matplotlib import pyplot

TRAINING_DIR = './training'

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

    frequency = {}
    for token in tokens:
        if token not in frequency:
            frequency[token] = 1
        else:
            frequency[token] += 1
    tokens = [x for x in tokens if frequency[x] > 1]
    
    return tokens


##
# Main method
##
def main():
    # Read in training files and format them for use in the model
    corpus = readFiles(TRAINING_DIR)
    corpus = [preprocess(x) for x in corpus]

    # Instantiate model
    model = models.Word2Vec(corpus, size=100, window=4, min_count=1, workers=6, alpha=0.025)
    
    # print(model)
    # print(model['JERRY:'])
    # print(model.wv['JERRY:'])

    # result = TruncatedSVD(n_components=2).fit_transform(model[model.wv.vocab])
    U, Sigma, VT = randomized_svd(model[model.wv.vocab], n_components=2)
    print("U", U)
    print("SIGMA", Sigma)
    print("VT", VT)
    # pyplot.scatter(result[:, 0], result[:, 1])
    # for i, word in enumerate(list(model.wv.vocab)):
    #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()


if __name__ == '__main__':
    main()
