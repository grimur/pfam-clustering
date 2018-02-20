import csv
import os
from aalist import AA_LIST

from run_model import load_model
from run_model import read_pfs
from run_model import word_count_vector


def predict_cluster_aa_prob(cluster, model, aa_specificities):
    # Predict the aa content of families in the model
    # cluster: List of BGC clusters as word vectors
    # model: The MM model
    # family: The family index to test
    # aa_specificities: List of probabilities for the given BGC ((aa, prob/cluster) pairs, where prob/cluster is the 
    #    probability that this aa exists in this cluster, ordered same as the cluster list.)
    model_results = []

    bgc_probability_matrix = model.predict_proba(cluster)

    for index in xrange(len(model.components)):
        # bgc_prob = model.predict_family(cluster, index)
        bgc_prob = [x[index] for x in bgc_probability_matrix]
        norm_prob = sum(bgc_prob)
        print norm_prob
        results = []
        for aa_id, aa_prob in aa_specificities.items():
            probability = sum([x * y for x, y in zip(bgc_prob, aa_prob)]) / norm_prob
            results.append((aa_id, probability))
        model_results.append(results)
    return model_results


def main():
    modelfile = '/home/grimur/data-crusemann/mixture-crusemann-mibig-justin-lcs/NRPS/allNRPS_clustering_c0.55.cluster.model'
    aa_dir = '/home/grimur/data-crusemann/aa-pred'
    bgc_dir = '/home/grimur/data-crusemann/bigscape-crusemann-mibig-justin-lcs/pfs_nrps/'

    # Leave BGCs with no predictied AAs out (incl. MIBiG)
    filter_zeros = True

    # load model, word list
    print 'Load model'
    vocabulary, model = load_model(modelfile)

    # load AA specificity predictions
    print 'Load AA specificity predictions'
    bgc_aa_prob = {}
    aa_index = []
    for filename in os.listdir(aa_dir):
        bgc_aa = [0.0] * len(aa_index)
        if (not filename.endswith('aa.csv')) or 'final' in filename:
            continue
        with open(os.path.join(aa_dir, filename)) as f:
            csvreader = csv.reader(f)
            for aa, prob in csvreader:
                prob = float(prob)
                if aa not in aa_index:
                    aa_index.append(aa)
                    bgc_aa.append(prob)
                else:
                    aa_i = aa_index.index(aa)
                    bgc_aa[aa_i] = prob
        if filter_zeros and sum(bgc_aa) == 0.0:
            continue
        else:
            bgc_id_fields = filename.split('.')
            bgc_id = '.'.join(bgc_id_fields[:-2])
            bgc_aa_prob[bgc_id] = bgc_aa

    # load BGCs
    # preprocess BGCs
    # todo: Only run on NRPS!!! (filter the PFS files in the input directory)
    print 'Load BGCs'
    bgc_collection = {}
    for filename in os.listdir(bgc_dir):
        # Only gbk files, skip files with 'final' in name
        if not filename.endswith('.pfs'):
            print filename
            continue
        sample = read_pfs(os.path.join(bgc_dir, filename))
        vector = word_count_vector(sample, vocabulary)
        bgc_id_fields = filename.split('.')
        bgc_id = '.'.join(bgc_id_fields[:-1])
        bgc_collection[bgc_id] = vector

    # Filter BGCs to AA predictions and vice versa
    print 'Merge AA and BGC predictions'
    bgc_list = []
    aa_predictions = {}
    for bgc_id, aa_prob in bgc_aa_prob.items():
        if bgc_id not in bgc_collection:
            continue
        bgc_vector = bgc_collection[bgc_id]
        bgc_list.append(bgc_vector)
        for aa_id, aa_prob in zip(aa_index, aa_prob):
            if aa_id in aa_predictions:
                aa_predictions[aa_id].append(aa_prob)
            else:
                aa_predictions[aa_id] = [aa_prob]

    print aa_predictions

    # Go!
    print 'Running actual predictions'
    pred = predict_cluster_aa_prob(bgc_list, model, aa_predictions)

    print pred
    with open('aa_pred_res.bin', 'wb') as f:
        import cPickle
        cPickle.dump(pred, f)


if __name__ == '__main__':
    main()
