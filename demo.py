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
    text = text.replace("'", " ' ")
    text = text.replace('"', ' " ')
    text = text.replace('(', ' ( ')
    text = text.replace(')', ' ) ')
    text = text.replace(':', ' : ')
    text = text.replace('[', ' [ ')
    text = text.replace(']', ' ] ')
    text = text.replace('!', ' !')
    text = text.replace('?', ' ?')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = text.lower()

    tokens = text.split()
    tokens = [x for x in tokens if x not in stopwords]
    
    return tokens

def train_model(corpus):
    # Instantiate empty model
    model = models.Word2Vec(min_count=1, iter=1, size=300)

    # Train model on data
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

    return model

def retrain_model(model, corpus):
    # Train model on new data
    model.build_vocab(corpus, update=True)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

    return model


def printModelStatistics(model):
    print("\tJerry-George similarity:", model.wv.n_similarity(['jerry'], ['george']))
    print("\tBig-Salad similarity", model.wv.n_similarity(['big'], ['salad']))
    print("\tWords most similar to 'Elaine':", sorted([x[0] for x in model.wv.most_similar('elaine')]))
    print()

##
# Main method
##
def main():
    STOPWORDS = [x.replace("\n", "") for x in readFile(STOPWORDS_FILE)]

    # Read in training files and format them for use in the model
    target_corpus = readFile(TRAINING_FILE)
    target_corpus.append("puffy shirt") # Have to add so these words are in the vocabulary for the example
    target_corpus = [preprocess(x, STOPWORDS) for x in target_corpus]
    # print(target_corpus)

    # Read in data that will be used to adapt the model
    source_corpus = readFile(ADAPT_FILE)
    source_corpus.append("salad")
    source_corpus = [preprocess(x, STOPWORDS) for x in source_corpus]
    # print(adaptation_corpus)


    # 1: Source-only method
    print("Source Only Method:")
    source_only = train_model(source_corpus)
    printModelStatistics(source_only)

    # 2: Source+target method
    print("Source+Target Method:")
    source_target = train_model(target_corpus+source_corpus)
    printModelStatistics(source_target)

    # 3: Weighted concatenate method
    print("Weighted-Concatenate Method:")
    source_target = train_model(target_corpus+source_corpus+source_corpus)
    printModelStatistics(source_target)
    

    # 4: Retrain source method
    print("Retrain Source Method:")
    retrain_source = train_model(target_corpus)
    retrain_source = retrain_model(retrain_source, source_corpus)
    printModelStatistics(retrain_source)


if __name__ == '__main__':
    print()
    main()
