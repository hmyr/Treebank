# -*- coding: utf-8 -*-

import sys, csv, time
from time import gmtime, strftime
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class DECluster(object):
    ids = []

    def __init__(self, fn, n_test_samples=1000, target_feat_name='childCheck', id_name='id', separator=','):
        self.target, self.features, self.feat_names, self.feat_ids = DECluster.read_features(fn)
        self.n_samples, self.n_features = self.features.shape
        self.n_digits = len(np.unique(self.target))
        self.sample_size = 300
        self.target = self.target[0:n_test_samples]
        self.features = self.features[0:n_test_samples]
        self.feat_names = self.feat_names[0:n_test_samples]
        self.feat_ids = self.feat_ids[0:n_test_samples]


    @classmethod
    def read_features(cls, fn,  target_feat_name='childCheck', id_name='id', separator=','):
        feats = []
        target_class = []
        ids = []
        with open(fn, 'rb') as infile:
            csv_reader = csv.DictReader(infile, delimiter=separator)
            feat_names = csv_reader.fieldnames
            for row in csv_reader:
                if row[target_feat_name]:
                    target_class.append(cls.convert_features(row[target_feat_name]))
                if row[id_name]:
                    ids.append(row[id_name])
                feats.append([cls.convert_features(row[fieldname]) for fieldname in feat_names
                              if fieldname != target_feat_name and fieldname != id_name])

        feat_names.remove(target_feat_name)
        feat_names.remove(id_name)
        return np.array(target_class, dtype=bool), np.array(feats, dtype=object), feat_names, ids

    @classmethod
    def convert_features(cls, item):
        if item == 'True': return True
        elif item == 'False': return False
        elif item.isdigit: return int(item)


    def log_it(self):
        sys.stderr.write(("\nn_targets: %d, \t n_samples %d, \t n_features %d"
                          % (self.n_digits, self.n_samples, self.n_features)))
        sys.stderr.write('\n%s' % str(79 * '_'))
        sys.stderr.write(('\n% 9s' % 'init\t'
                          '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette'))

    def bench_k_means(self, estimator, name, data):
        t0 = time.time()
        estimator.fit(data)
        sys.stderr.write(('\n% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
                          % (name, ( time.time() - t0), estimator.inertia_,
                            metrics.homogeneity_score(self.target, estimator.labels_),
                            metrics.completeness_score(self.target, estimator.labels_),
                            metrics.v_measure_score(self.target, estimator.labels_),
                            metrics.adjusted_rand_score(self.target, estimator.labels_),
                            metrics.adjusted_mutual_info_score(self.target,  estimator.labels_),
                            metrics.silhouette_score(data, estimator.labels_,
                                                     metric='euclidean',
                                                     sample_size=self.sample_size))))

    def clusterize(self, init=None, n_init=10):
        self.log_it()
        self.bench_k_means(KMeans(init='k-means++', n_clusters=self.n_digits, n_init=10),
              name="k-means++", data=self.features)

        self.bench_k_means(KMeans(init='random', n_clusters=self.n_digits, n_init=10),
              name="random", data=self.features)

        pca = PCA(n_components=self.n_digits).fit(self.features)
        self.bench_k_means(KMeans(init=pca.components_, n_clusters=self.n_digits, n_init=1),
              name="PCA-based",
              data=self.features)
        sys.stderr.write('\n%s' % 79 * '_')

        self.reduced_data = PCA(n_components=2).fit_transform(self.features)
        self.kmeans = KMeans(init='k-means++', n_clusters=self.n_digits, n_init=10)
        self.kmeans.fit(self.reduced_data)

    def plot_clusters(self):
        h = .02
        x_min, x_max = self.reduced_data[:, 0].min() + 1, self.reduced_data[:, 0].max() - 1
        y_min, y_max = self.reduced_data[:, 1].min() + 1, self.reduced_data[:, 1].max() - 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        plt.plot(self.reduced_data[:, 0], self.reduced_data[:, 1], 'k.', markersize=2)
        centroids = self.kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10)
        plt.title('K-means clustering on the RSTB markup  (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


if __name__ == '__main__':
    in_fn = sys.argv[1]

    decluster = DECluster(in_fn)
    decluster.clusterize()
    decluster.plot_clusters()




