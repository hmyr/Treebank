# -*- coding: utf-8 -*-

import sys, csv
from time import gmtime, strftime
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold
import logging


csv.field_size_limit(sys.maxint)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DECluster(object):
    ids = []

    def __init__(self, fn, n_test_samples=300,  n_clusters=5):
        self.target, self.features, self.feat_names, self.feat_ids = DECluster.read_features(fn)
        self.n_samples, self.n_features = self.features[0:n_test_samples].shape
        self.n_clusters = n_clusters
        self.sample_size = 3

    @classmethod
    def read_features(cls, fn,  target_feat_name='childCheck', id_name='id', default_target_value=False):
        """
        'id' reads as follows: head.SID, head.WID, child.WID)
        :param fn:
        :param target_feat_name:
        :param id_name:
        :param separator:
        :return:
        """
        feats = []
        target_class = []
        ids = []
        default_delimiter = ';'
        with open(fn, 'rb') as infile:
            try:
                dialect = csv.Sniffer().sniff(infile.read(1024))
                delimiter = dialect.delimiter
                infile.seek(0)
                csv_reader = csv.DictReader(infile, delimiter=delimiter)
            except Exception, error:
                logging.error('{}. Trying default delimiter...'.format(error))
                delimiter = default_delimiter
                csv_reader = csv.DictReader(infile, delimiter=delimiter)
            feat_names = csv_reader.fieldnames
            if target_feat_name in feat_names:
                for row in csv_reader:
                    if row[target_feat_name]:
                        target_class.append(cls.convert_features(row[target_feat_name]))
                    if row[id_name]:
                        ids.append(row[id_name])
                    feats.append([cls.convert_features(row[fieldname]) for fieldname in feat_names
                                 if fieldname != target_feat_name and fieldname != id_name])
                feat_names.remove(target_feat_name)
            else:
                for row in csv_reader:
                    target_class.append(default_target_value)
                    if id_name in row:
                        ids.append(row[id_name])
                    feats.append([cls.convert_features(row[fieldname]) for fieldname in feat_names
                                  if fieldname != target_feat_name and fieldname != id_name])

        feat_names.remove(id_name)
        return np.array(target_class, dtype=bool), np.array(feats, dtype=object), feat_names, ids

    @classmethod
    def convert_features(cls, item):
        if item == 'True': return True
        elif item == 'False': return False
        elif item.isdigit: return int(item)

    def hierarchical_clustering(self, linkage='ward'):

        sys.stdout.write('\nComputing embedding... at %s\n' % strftime("%a, %d %b %Y %H:%M:%S\n", gmtime()))
        self.features_embedded = manifold.SpectralEmbedding(n_components=2).fit_transform(self.features)
        sys.stdout.write('\nDone... at %s\n' % strftime("%a, %d %b %Y %H:%M:%S\n", gmtime()))

        self.clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
        sys.stdout.write('\nPerforming agglomerative clustering... at %s\n' %
                         strftime("%a, %d %b %Y %H:%M:%S\n", gmtime()))
        self.clustering.fit(self.features_embedded)
        sys.stdout.write('\nDone... at %s' % strftime("%a, %d %b %Y %H:%M:%S\n", gmtime()))
        # self.save_clustering_tree(outfn=outfn)

    def save_clustering_tree(self, outfn):
        headers = ['cluster', 'id', 'parent']
        with open(outfn, 'wb') as outfile:
            outfile.write(';'.join(headers) + '\n')
            for node_id, x in enumerate(self.clustering.children_):
                left_child = x[0]
                right_child = x[1]
                cluster_id = self.clustering.labels_[node_id]
                outfile.write('\n%s;%s;%s' % (cluster_id, left_child, node_id))
                outfile.write('\n%s;%s;%s' % (cluster_id, right_child, node_id))

    def process_clustering_tree(self):
        for node_id, x in enumerate(self.clustering.children_):
            left_child = x[0]
            lch_id = self.feat_names[left_child]
            right_child = x[1]
            rch_id = self.feat_names[right_child]
            cluster_id = self.clustering.labels_[node_id]
            node_id = self.feat_names[node_id]

if __name__ == '__main__':
    in_fn = sys.argv[1]
    outdir = sys.argv[2]
    n_clusters = 3
    n_test_samples = 300
    outfn = '%sclusterized_features_%s_n_test_samples_Hierarch.csv' % (outdir,  n_test_samples)
    decluster = DECluster(in_fn, n_clusters=n_clusters, n_test_samples=n_test_samples)
    decluster.hierarchical_clustering()
    decluster.process_clustering_tree()





