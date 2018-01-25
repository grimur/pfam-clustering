from __future__ import print_function

from run_model import read_pfs
from run_model import word_count_vector
from estimate_overlap import parse_individual_cluster, estimate_overlap
from cluster_bgcs import parse_list
from cluster_bgcs import preprocess_list
from cluster_bgcs import heatmap
from cluster_bgcs import build_cluster_model
from cluster_bgcs import test_cluster_model
from cluster_bgcs import read_from_family_file

import argparse

import collections
import cPickle
import numpy
import os
import pandas
import sys


def extract_family_data(model, data, number):
    filtered_data = []
    for index, line in data.iterrows():
        prediction = model.predict(line.values.reshape(1, -1))
        if prediction == number:
            filtered_data.append(index.split('.')[0])
    return filtered_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-base', help='base directory', dest='basedir', required=True)
    parser.add_argument('-familyfile', help='family file', dest='familyfile', required=True)
    parser.add_argument('-pfs', help='pfs directory', dest='pfsdir', required=True)
    parser.add_argument('-dest', help='destination directory', dest='dest', required=True)

    args = parser.parse_args()

    family_file = os.path.join(args.basedir, args.familyfile)
    pfs_dir = os.path.join(args.basedir, args.pfsdir)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    print('Parsing PFS- and family files...'),
    df = parse_list(read_from_family_file(family_file, pfs_dir))
    print('done.')

    filename = args.familyfile.split(os.path.sep)[-1]
    filename_segments = filename.split('.')
    if len(filename_segments) > 1:
        filename_segments.pop(-1)
    filename = '.'.join(filename_segments)

    graph_output = os.path.join(args.dest, '%s.cluster.png' % filename)
    df_output = os.path.join(args.dest, '%s.cluster.bin' % filename)
    # report_output = os.path.join(args.dest, '%s.cluster.report.txt' % filename)
    family_output = os.path.join(args.dest, '%s.families.tsv' % filename)
    model_output = os.path.join(args.dest, '%s.cluster.model' % filename)

    df = df.set_index('id')
    df.reindex_axis(sorted(df.columns), axis=1)
    # heatmap(df, graph_output)

    print('Writing word count data...'),
    with open(df_output, 'wb') as f:
        cPickle.dump(df, f)
    print('done.')

    model_type = 'multinomial'
    print('Building %s model...' % model_type),
    cluster_model = build_cluster_model(df, method=model_type)
    # cluster_model = build_cluster_model(df, method='multinomial')
    print('done.')
    vocabulary = list(df.columns)

    with open(model_output, 'wb') as f:
        cPickle.dump((vocabulary, cluster_model), f)

    # Test model against bigscape.
    # Might not be such a hot idea - maybe only use MiBIG-tagged clusters?
    # And here is def. not the place to do it.
    # test_cluster_model(cluster_model, df, report_output, family_output)
