from __future__ import print_function

from sklearn.mixture import GaussianMixture
from mm import MixtureModel as MultinomialMixture

import argparse

import collections
import cPickle
import os
import pandas
import sys


def parse_list(iterable):
    """
    Input is assumed to be of form
      (id, "word1 word2 word3 ...")
    i.e. a tuple of id and space separated word list.

    Returns a data frame with word count as columns, plus ID column.
    """
    # Create an initial empty data frame
    df = pandas.DataFrame(data=[], columns=['id'])

    # add each line
    for line_id, words in iterable:
        words = words.split(' ')

        # create the row that will be appended to the data frame
        word_count_line = [0] * len(df.columns)
        word_count_line[df.columns.get_loc('id')] = line_id

        for word in words:
            word = word.strip()
            # Add a new column if the word isn't in dictionary already
            if word not in df.columns:
                # add column to frame
                df[word] = [0] * len(df)
                word_count_line.append(0)

            # find word index and increment
            word_index = df.columns.get_loc(word)
            word_count_line[word_index] += 1

        # append line to data frame
        word_count_line = pandas.DataFrame([word_count_line], columns=list(df.columns))
        df = df.append(word_count_line, ignore_index=True)

    return df


def preprocess_list(iterable):
    """
    Take an iterable yielding lines of form
      'idA, idB, contents'
    and convert to tuples
      ('idA:idB', contents)
    """
    for line in iterable:
        strain_id, cluster_id, words = line.split(',')
        line_id = "{}:{}".format(strain_id, cluster_id)
        yield line_id, words


def heatmap(df, output=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plot = sns.heatmap(df)
    if output is None:
        plt.show()
    else:
        figure = plot.get_figure()
        figure.savefig(output)


def build_cluster_model(df, method='gaussian'):
    # estimate number of clusters (i.e. number of proper families)
    families = set([])
    for pfam_id in df.index:
        pfam, family = pfam_id.split(':')
        families.add(family)

    num_clusters = len(families)
    if method is 'gaussian':
        cluster_model = GaussianMixture(n_components=num_clusters)
    elif method is 'multinomial':
        cluster_model = MultinomialMixture(n_components=num_clusters)
    else:
        raise ValueError('Unknown model %s' % method)
    cluster_model.fit(df)

    return cluster_model


def test_cluster_model(model, data, report_file=None, family_file=None):
    predictions = []
    original_families = set([])
    for index, line in data.iterrows():
        bgc_id, bigscape_family = index.split(':')
        original_families.add(bigscape_family)
        predicted_family = model.predict(line.values.reshape(1, -1))
        predictions.append((bgc_id, bigscape_family, predicted_family[0]))
        # print((bgc_id, bigscape_family, predicted_family))

    match_count = 0
    total_count = 0
    for bigscape_family in original_families:
        classified_families = [x[2] for x in predictions if x[1] == bigscape_family]
        total_count += len(classified_families)
        if len(set(classified_families)) == 1:
            match_count += len(classified_families)
        else:
            class_count = collections.Counter(classified_families)
            family_id, count = class_count.most_common()[0]
            match_count += count

    output = 'Total matches: {}/{} ({}%)'.format(match_count, total_count, float(match_count) / total_count)
    if report_file is None:
        print(output)
    else:
        with open(report_file, 'w') as f:
            f.write(output + '\n')

    if family_file is not None:
        with open(family_file, 'w') as f:
            predicted_families = set([x[2] for x in predictions])
            for family in predicted_families:
                bgcs = [x[0] for x in predictions if x[2] == family]
                for bgc in bgcs:
                    f.write('%s\t%s\n' % (bgc, family))


def read_from_family_file(family_file, pfs_dir):
    """
    Read families in tab separated <cluster_id>, <family> pairs from family_file
    Looks for <cluster_id>.pfs in pfs_dir for the pfam contents of the cluster.
    Yields pairs of <cluster_id>:<family_id>, "<pfam1> <pfam2> ..."
    """
    with open(family_file, 'r') as f:
        cnt = 0
        limit = 0  # cnt > 0 for all iterations
        for cluster_line in f.readlines():
            cnt += 1
            if cnt == limit:
                break
            if cluster_line.startswith('#'):
                continue
            cluster_id, family_id = cluster_line.strip().split()
            cluster_pfs_file = os.path.join(pfs_dir, '%s.pfs' % cluster_id)
            with open(cluster_pfs_file, 'r') as pfs_f:
                pfs_string = pfs_f.read().strip()
                pfs_list = pfs_string.split()
                pfs_roots = [x.split('.')[0] for x in pfs_list]
                pfs_root_string = ' '.join(pfs_roots)
            yield '%s:%s' % (cluster_id, family_id), pfs_root_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-base', help='base directory', dest='basedir')
    parser.add_argument('-familyfile', help='family file', dest='familyfile')
    parser.add_argument('-pfs', help='pfs directory', dest='pfsdir')
    parser.add_argument('-dest', help='destination directory', dest='dest')

    args = parser.parse_args()

    family_file = os.path.join(args.basedir, args.familyfile)
    pfs_dir = os.path.join(args.basedir, args.pfsdir)

    df = parse_list(read_from_family_file(family_file, pfs_dir))

    filename = args.familyfile.split(os.path.sep)[-1]
    filename_segments = filename.split('.')
    if len(filename_segments) > 1:
        filename_segments.pop(-1)
    filename = '.'.join(filename_segments)

    graph_output = os.path.join(args.dest, '%s.cluster.png' % filename)
    df_output = os.path.join(args.dest, '%s.cluster.bin' % filename)
    report_output = os.path.join(args.dest, '%s.cluster.report.txt' % filename)
    family_output = os.path.join(args.dest, '%s.families.tsv' % filename)
    model_output = os.path.join(args.dest, '%s.cluster.model' % filename)

    with open(df_output, 'wb') as f:
        cPickle.dump(df, f)

    df = df.set_index('id')
    df.reindex_axis(sorted(df.columns), axis=1)
    # heatmap(df, graph_output)

    cluster_model = build_cluster_model(df)
    vocabulary = list(df.columns)

    with open(model_output, 'wb') as f:
        cPickle.dump((vocabulary, cluster_model), f)

    test_cluster_model(cluster_model, df, report_output, family_output)

