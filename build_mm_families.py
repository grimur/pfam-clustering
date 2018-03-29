from __future__ import print_function

import argparse
import cPickle
import os
import sklearn


def load_model(filename):
    with open(filename, 'rb') as f:
        vocabulary, model = cPickle.load(f)
    return vocabulary, model


def read_pfs(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data.strip().split()


def word_count_vector(sentence, vocabulary, quiet=False):
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

    if not quiet:
        print('Vocabulary size: %s' % vocabulary_size)
        print('Sentence vocabulary size: %s' % sentence_vocabulary_size)
        print('Sentence length: %s' % sentence_length)
        print('Skipped words: %s' % skipped_words)

    return output_vector


def parse_family_file(path):
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            bgc_id, family = line.strip().split()
            yield bgc_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='modelfile', help='Input model file', required=True)
    parser.add_argument('--families', dest='familyfile', help='bigscape family file', required=True)
    parser.add_argument('--pfs', dest='pfsdir', help='pfs file path', required=True)
    parser.add_argument('--prob', dest='prob', help='Output classification probabilities', default=False, action='store_true')
    parser.add_argument('-f', dest='filter', help='Filter probabilities (only output >0)', default=False, action='store_true')

    args = parser.parse_args()

    # print('Loading model from %s' % args.modelfile)
    vocabulary, model = load_model(args.modelfile)

    # print('Loading BGC entries from %s' % args.familyfile)
    bgc_entries = parse_family_file(args.familyfile)

    for bgc_id in bgc_entries:
        bgc_path = os.path.join(args.pfsdir, bgc_id)
        bgc_path = '%s.pfs' % bgc_path
        bgc_sample = read_pfs(bgc_path)

        bgc_vector = word_count_vector(bgc_sample, vocabulary, quiet=True)

        if args.prob:
            results = model.predict_proba([bgc_vector])[0]
            for family, prob in enumerate(results):
                if not (prob == 0 and args.filter):
                    print('%s\t%s\t%s' % (bgc_id, family, prob))
        else:
            results = model.predict_with_proba([bgc_vector])[0]
            print('%s\t%s' % (bgc_id, results[0]))


if __name__ == '__main__':
    main()
