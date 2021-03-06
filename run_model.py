from __future__ import print_function

import argparse
import cPickle
import sklearn


def load_model(filename):
    with open(filename, 'rb') as f:
        vocabulary, model = cPickle.load(f)
    return vocabulary, model


def read_pfs(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data.strip().split()


def word_count_vector(sentence, vocabulary):
    output_vector = [0] * len(vocabulary)
    vocabulary_size = len(vocabulary)
    sentence_vocabulary_size = len(set(sentence))
    sentence_length = len(sentence)
    skipped_words = 0
    for word in sentence:
        word = word.split('.')[0]
        if word not in vocabulary:
            skipped_words += 1
            continue
        word_index = vocabulary.index(word)
        output_vector[word_index] += 1

    print('Vocabulary size: %s' % vocabulary_size)
    print('Sentence vocabulary size: %s' % sentence_vocabulary_size)
    print('Sentence length: %s' % sentence_length)
    print('Skipped words: %s' % skipped_words)

    return output_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='modelfile', help='Input model file', required=True)
    parser.add_argument('-f', dest='inputfile', help='Pfam domains to classify', required=True)
    parser.add_argument('-v', dest='verbose', help='Output all families', required=False, default=False, action='store_true')

    args = parser.parse_args()

    print('Loading model...')
    vocabulary, model = load_model(args.modelfile)
    print('Done.')
    sample = read_pfs(args.inputfile)

    vector = word_count_vector(sample, vocabulary)

    if args.verbose:
        # only have one vector, so...
        results = model.predict_proba([vector])[0]
    else:
        results = model.predict([vector])[0]

    print(results)
    for idx, res in enumerate(results):
        if res > 0:
            print(idx, res)


if __name__ == '__main__':
    main()
