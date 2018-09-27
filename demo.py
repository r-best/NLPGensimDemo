import os
import multiprocessing
from gensim import corpora, models

# Training file to be used for initial training
TRAINING_FILE = './training/6-2 The Big Salad'

# Training file to be used for adaptation
ADAPT_FILE = './training/5-2 The Puffy Shirt'

STOPWORDS_FILE = './stopwords.txt'

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
# Takes in a string, applies a set of 
# preprocessing rules to it, and splits
# it into a list of its tokens
##
def preprocess(text, stopwords):
    text = text.replace('\n', ' \n ')
    text = text.replace('(', ' ( ')
    text = text.replace(')', ' ) ')
    text = text.replace(':', ' : ')
    text = text.replace('[', ' [ ')
    text = text.replace(']', ' ] ')
    text = text.replace('!', ' !')
    text = text.replace('?', ' ?')
    text = text.replace('.', ' .')
    text = text.lower()

    tokens = text.split()
    tokens = [x for x in tokens if x not in stopwords]
    
    return tokens


##
# Main method
##
def main():
    STOPWORDS = [x.replace("\n", "") for x in readFile(STOPWORDS_FILE)]

    # Read in training files and format them for use in the model
    initial_corpus = readFile(TRAINING_FILE)
    initial_corpus.append("puffy shirt") # Have to add so these words are in the vocabulary for the example
    initial_corpus = [preprocess(x, STOPWORDS) for x in initial_corpus]
    # print(initial_corpus)

    # Read in data that will be used to adapt the model
    adaptation_corpus = readFile(ADAPT_FILE)
    adaptation_corpus = [preprocess(x, STOPWORDS) for x in adaptation_corpus]
    # print(adaptation_corpus)

    # Instantiate empty model
    model = models.Word2Vec(min_count=1, iter=1, size=300)

    # Train model on initial data
    model.build_vocab(initial_corpus)
    model.train(initial_corpus, total_examples=model.corpus_count, epochs=model.iter)
    print("Initial Model:", model, '\n')
    print("Initial similarity between 'big' and 'salad':", model.wv.n_similarity(['big'], ['salad']))
    print("Adapted similar words to 'big':", sorted([x[0] for x in model.wv.most_similar('big')]))
    print("Adapted similar words to 'puffy':", sorted([x[0] for x in model.wv.most_similar('puffy')]))

    print()

    # Train model with some new data
    model.build_vocab(adaptation_corpus, update=True)
    model.train(adaptation_corpus, total_examples=model.corpus_count, epochs=model.iter)
    # print("Adapted Model:", model, '\n')
    print("Adapted similarity between 'big' and 'salad':", model.wv.n_similarity(['big'], ['salad']))
    print("Adapted similar words to 'big':", sorted([x[0] for x in model.wv.most_similar('big')]))
    print("Adapted similar words to 'puffy':", sorted([x[0] for x in model.wv.most_similar('puffy')]))


if __name__ == '__main__':
    print()
    main()
