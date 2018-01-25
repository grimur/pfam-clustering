import argparse
import cPickle
import numpy
import scipy.stats


def estimate_overlap(set_1, set_2, total_size, test='pearson', parameters=None):
    """
    Support pearson or hypergeometric
    set_1 = bgc family strains
    set_2 = mf strains
    """
    if test == 'pearson':
        # pearson correlation test.
        # Pros: simple. Paired.
        # Cons: Symmetric, i.e. does not account for crypticity
        all_ids = set(set_1 + set_2)
        vector_1 = []
        vector_2 = []
        for i in all_ids:
            if i in set_1:
                vector_1.append(1)
            else:
                vector_1.append(0)
            if i in set_2:
                vector_2.append(1)
            else:
                vector_2.append(0)
        vector_1.extend([0] * (total_size - len(vector_1)))
        vector_2.extend([0] * (total_size - len(vector_2)))

        correlation, value = scipy.stats.pearsonr(vector_1, vector_2)
        value = (correlation, value)

    elif test == 'hypergeometric':
        # Hypergeometric test with set_2 as ground truth.
        # Pros: Intuitive?
        # Cons: Symmetric? Not properly paired? Does not filter in crypticity?
        # N - total number
        # K - total positives in population
        # n - sample size
        # k - positives in sample
        k = len(set(set_1).intersection(set(set_2)))
        K = len(set_1)
        n = len(set_2)
        N = total_size

        value = scipy.stats.hypergeom.sf(k-1, N, n, K)

    elif test == 'generative':
        # Generative probablistic model
        # Pros: Intuitive, tuneable
        # Cons: Need to set noise and p(cryptic)
        set_1 = set(set_1)
        set_2 = set(set_2)

        if parameters is not None and 'p_cryptic' in parameters:
            p_cryptic = parameters['p_cryptic']
        else:
            p_cryptic = 0.25
        if parameters is not None and 'p_noise' in parameters:
            p_noise = parameters['p_noise']
        else:
            p_noise = 0.01

        intersection_count = len(set_1.intersection(set_2))
        cryptic_count = len(set_2 - set_1)
        noise_count = len(set_1 - set_2)
        true_negatives = total_size - (intersection_count + cryptic_count + noise_count)

        value = (1 - p_cryptic) ** intersection_count \
            * p_cryptic ** cryptic_count \
            * p_noise ** noise_count \
            * (1 - p_noise) ** true_negatives

    elif test == 'confusion':
        set_1 = set(set_1)
        set_2 = set(set_2)

        tp = len(set_1.intersection(set_2))
        fn = 0
        fp = len(set_2 - set_1)
        tn = total_size - (tp + fp + fn)
        
        if tp == 0 or tn == 0:
            value = 0
        else:
            confusion_matrix = [[tp, fp],
                                [fn, tn]]

            # Matthews correlation coefficient
            value = ((tp * tn) - (fp * fn)) / numpy.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return value


def parse_family_file(filename):
    gcf_dict = {}

    with open(filename, 'r') as f:
        for line in f.readlines():
            cluster_id, family_id = line.strip().split()
            strain_id = cluster_id.split('.')[0]

            if family_id in gcf_dict:
                gcf_dict[family_id].append(strain_id)
            else:
                gcf_dict[family_id] = [strain_id]

    return gcf_dict


def parse_individual_cluster(filename):
    strains = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if filename.endswith('.pfs'):
                strains.extend(line.strip().split())
            else:
                strains.append(line.strip())
    return strains


def run_comparison_family_vs_cluster(family_file, cluster_file, pool_size, method):
    families = parse_family_file(family_file)
    external_cluster = parse_individual_cluster(cluster_file)

    for family_id, family in families.items():
        value = estimate_overlap(family, external_cluster, pool_size, method)
        print family_id, value


def run_comparison_cluster_vs_cluster(cluster_file_1, cluster_file_2, pool_size, method):
    cluster_1 = parse_individual_cluster(cluster_file_1)
    cluster_2 = parse_individual_cluster(cluster_file_2)

    value = estimate_overlap(cluster_1, cluster_2, pool_size, method)
    print value


def read_pool_size(filename):
    count = 0
    with open(filename, 'r') as f:
        for l in f.readlines():
            count += 1
    return count


def load_binary(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data


def run_binary_comparisons(binary_files, pool_size):
    supported_methods = ['pearson', 'hypergeometric', 'generative', 'confusion']

    family_pairs = [load_binary(x) for x in binary_files]

    for method in supported_methods:
        results = []
        for i in xrange(len(family_pairs)):
            for j in xrange(len(family_pairs)):
                bgcfamily = family_pairs[i][1]
                bgcid = family_pairs[i][0]
                msfamily = family_pairs[j][2]
                msid = family_pairs[j][0]
                comp_value = estimate_overlap(bgcfamily, msfamily, pool_size, method)
                results.append((method, bgcid, msid, comp_value))
        results.sort(key=lambda x: x[3])
        for a in results:
            print a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='family_file', help='Family .tsv file', nargs='+', default=[])
    parser.add_argument('-c', dest='cluster_file', help='Cluster(s) to compare', nargs='+', default=[])
    parser.add_argument('-n', dest='pool', help='ID pool. Number or file', required=True)
    parser.add_argument('-m', dest='method', help='Scoring method, defult pearson', default='pearson')
    parser.add_argument('-b', dest='binary', help='Binary file(s)', nargs='+', default=[])

    args = parser.parse_args()

    try:
        pool_size = int(args.pool)
    except ValueError:
        pool_size = read_pool_size(args.pool)

    method = args.method

    if len(args.cluster_file) == 2:
        cf1, cf2 = args.cluster_file
        run_comparison_cluster_vs_cluster(cf1, cf2, pool_size, method)
    elif len(args.family_file) == 2:
        raise SystemExit('not implemented yet')
    elif len(args.family_file) == 1 and len(args.cluster_file) == 1:
        cf = args.cluster_file[0]
        ff = args.family_file[0]
        run_comparison_family_vs_cluster(ff, cf, pool_size, method)

    elif len(args.binary) > 0:
        run_binary_comparisons(args.binary, pool_size)

    if len(args.cluster_file) + len(args.family_file) != 2 and len(args.binary) == 0:
        raise SystemExit('Please use total of two files')


if __name__ == '__main__':
    main()
