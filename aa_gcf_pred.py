import argparse
import csv
import numpy
import os

from aalist import AA_LIST
import aa_pred



_STRAIN = False


def predict_gcf_aa_prob(families, aa_pred, aa_list):
    """
    Predict the aa content of families.
    families: a dict of family_id : [gcfs]
    aa_pred: a dict of bgc_id : [aa_p]
    aa_list: a list of the amino acids
    """
    gcf_aa_probs = {}
    max_prob = 0

    for gcf_id, gcf_clusters in families.items():
        fam_aa = {}
        for bgc_id in gcf_clusters:
            try:
                bgc_aa_pred = aa_pred[bgc_id]
            except KeyError:
                print 'Specificity predictions for %s not found' % bgc_id
                bgc_aa_pred = []
            for p_aa, aa_id in zip(bgc_aa_pred, aa_list):
                if aa_id in fam_aa:
                    fam_aa[aa_id].append(p_aa)
                else:
                    fam_aa[aa_id] = [p_aa]
        aa_probs = []
        for aa_id in aa_list:
            if aa_id in fam_aa:
                aa_probs.append(numpy.mean(fam_aa[aa_id]))
                if numpy.mean(fam_aa[aa_id]) > max_prob:
                    max_prob = numpy.mean(fam_aa[aa_id])
            else:
                aa_probs.append(0.0)

        if not sum(aa_probs) == 0.0:
            gcf_aa_probs[gcf_id] = aa_probs

    return gcf_aa_probs


def find_bgc_gbk_files(gbk_path, families):
    """
    Only load gbk files that belong to an interesting cluster.
    This automatically limits us to e.g. NRPS, because BiG-SCAPE has done
    that filtering for us.
    """
    bgcs = set([])
    for fam_id, bgc_ids in families.items():
        bgcs = bgcs.union(bgc_ids)

    bgc_files = []
    for filename in os.listdir(gbk_path):
        if not filename.endswith('.gbk'):
            continue
        for bgc_id in bgcs:
            if bgc_id in filename:
                bgc_files.append(os.path.join(gbk_path, filename))
                break

    return bgc_files


def build_bgc_aa_predictions(gbk_files, filter_zeros):
    """
    Build AA predictions for BGCs from gbk path
    """
    bgc_aa_prob = {}
    aa_index = []
    for filepath in gbk_files:
        # print 'processing %s' % filepath
        bgc_aa = [0.0] * len(aa_index)
        # Do the actual prediction
        for aa, prob in aa_pred.predict_aa(filepath):
            if aa not in aa_index:
                aa_index.append(aa)
                bgc_aa.append(prob)
            else:
                aa_i = aa_index.index(aa)
                bgc_aa[aa_i] = prob
        if filter_zeros and sum(bgc_aa) == 0.0:
            continue
        else:
            filename = os.path.basename(filepath)
            bgc_id_fields = filename.split('.')
            bgc_id = '.'.join(bgc_id_fields[:-1])
            bgc_aa_prob[bgc_id] = bgc_aa

    return bgc_aa_prob, aa_index


def load_family_file(file_path):
    """
    Load BiG-SCAPE .tsv family file
    """
    families = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            bgc_id, fam_id = line.strip().split()
            if fam_id in families:
                families[fam_id].append(bgc_id)
            else:
                families[fam_id] = [bgc_id]
    return families


def main(bgc_gbk_dir, gcf_file, filter_zeros):
    # print 'Load family file'
    gcf_bgcs = load_family_file(gcf_file)

    # print 'Find gbk files'
    bgc_gbk_files = find_bgc_gbk_files(bgc_gbk_dir, gcf_bgcs)

    # print 'build AA specificity predictions'
    bgc_aa_prob, aa_index = build_bgc_aa_predictions(bgc_gbk_files, filter_zeros)

    pred = predict_gcf_aa_prob(gcf_bgcs, bgc_aa_prob, aa_index)

    for gcf_id, aa_prob in pred.items():
        for aa_id, p_aa in zip(aa_index, aa_prob):
            print "%s\t%s\t%s" % (gcf_id, aa_id, p_aa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Predict AA specificity for BiG-SCAPE GCF')
    parser.add_argument('bgc', help='BGC .gbk file directory (collected antiSMASH output)')
    parser.add_argument('gcf', help='GCF file (BiG-SCAPE output)')
    parser.add_argument('--with_zeros', dest='filter_zeros', help='Include zero-predictions (e.g. MIBiG clusters)', action='store_false', default=True)
    args = parser.parse_args()

    # Leave BGCs with no predictied AAs out (incl. MIBiG)
    filter_zeros = args.filter_zeros

    bgc_gbk_dir = args.bgc
    gcf_file = args.gcf

    main(bgc_gbk_dir, gcf_file, filter_zeros)
