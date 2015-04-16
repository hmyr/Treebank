# -*- coding: utf-8 -*-

import cPickle, sys, csv
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, \
    ExtraTreesClassifier, GradientBoostingClassifier

from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.cross_validation import Bootstrap
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report,  confusion_matrix,
                             precision_score, make_scorer, recall_score)
from time import gmtime, strftime
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from clustering import DECluster


n_estimators = 100
n_b_estimators = 10


class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


class GradientBoostingClassifierWithCoef(GradientBoostingClassifier):
    def fit(self, *args, **kwargs):
        super(GradientBoostingClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_


classifiers = {'forest': RandomForestClassifier(n_estimators=n_estimators, min_samples_split=2, n_jobs=1),
               'forest_with_coef': RandomForestClassifierWithCoef(n_estimators=10, min_samples_leaf=5,
                                                                  min_samples_split=2, n_jobs=-1),
               'tree': tree.DecisionTreeClassifier(),
               'forest_bagging': BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=n_estimators,
                                                                                         min_samples_split=2, n_jobs=1),
                                                   n_estimators=n_b_estimators, max_samples=1.0, max_features=1.0,
                                                   bootstrap=True, bootstrap_features=True, oob_score=False, n_jobs=1,
                                                   random_state=True, verbose=0),
               'gradient_bagging': BaggingClassifier(base_estimator=GradientBoostingClassifier(loss='deviance',
                                                     learning_rate=0.1, n_estimators=n_estimators, subsample=1.0,
                                                     min_samples_split=2, min_samples_leaf=1, max_depth=3),
                                                     n_estimators=n_b_estimators, max_samples=1.0, max_features=1.0,
                                                     bootstrap=True, bootstrap_features=True, oob_score=False, n_jobs=1,
                                                     random_state=True, verbose=0),
               'bootstrap': Bootstrap(10013, n_iter=3, train_size=0.49, test_size=0.49, random_state=None,
                                      n_bootstraps=None),
               'feature_selection': Pipeline([
                   ('feature_selection', LinearSVC(penalty="l1", dual=False)),
                   ('classification', RandomForestClassifierWithCoef(n_estimators=n_estimators, min_samples_split=2,
                                                                     n_jobs=1))]),
               'linear_svc': LinearSVC(penalty="l1", dual=False),
               'extra_tree': ExtraTreesClassifier(n_estimators=n_estimators,  random_state=0),
               'gradient': GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=n_estimators,
                                                      subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                                                      max_depth=3, init=None, random_state=None, max_features=None,
                                                      verbose=0, max_leaf_nodes=None, warm_start=False),
               'regression_tree': tree.DecisionTreeRegressor(),
               'decision_bagging': tree.DecisionTreeClassifier(),
               'regression_bagging': tree.DecisionTreeRegressor(max_depth=2)
               }


class DEFinder(object):
    def __init__(self, fn, separator=',', target_feat_name='childCheck', id_name='id'):
        self.classifier = None
        self.target, self.features, self.feat_names, self.ids = \
            DECluster.read_features(fn, target_feat_name=target_feat_name, separator=separator, id_name=id_name)
        self.bootstrap_informative_features = dict()
        self.false_precision_scorer = make_scorer(precision_score, labels=None, pos_label=0, average='weighted')
        self.true_precision_scorer = make_scorer(precision_score, labels=None, pos_label=1, average='weighted')
        self.false_recall_scorer = make_scorer(recall_score, labels=None, pos_label=0, average='weighted')
        self.true_recall_scorer = make_scorer(recall_score, labels=None, pos_label=1, average='weighted')
        self.std = None

    def train_model(self, classifier, bootstrap=False):
        if classifier not in classifiers: return False
        self.classifier = classifiers[classifier]
        if bootstrap is False:
            self.classifier.fit(self.features, self.target)

        elif bootstrap is True:
            self.bootstrap()

    def get_important_feats(self):
        if hasattr(self.classifier, 'feature_importances_'):
                self.feature_importances_ = self.classifier.feature_importances_
        else:
            self.feature_importances_ = np.mean([tree.feature_importances_ for
                                                        tree in self.classifier.estimators_], axis=0)


    def rank_features(self, how_many=110):
        self.get_important_feats()
        self.indices = np.argsort(self.feature_importances_)[::-1]
        self.ranked_features_dict = dict([(self.feat_names[self.indices[f]],
                                             np.round(self.feature_importances_[self.indices[f]], 5))
                                             for f in range(how_many)])

    def show_most_informative_features(self, threshold=0.0):
        sys.stdout.write("Feature ranking:\n")
        sys.stdout.write('\n'.join('%s. %s - %s' % (n, i, self.ranked_features_dict[i])
                                   for n, i in enumerate(sorted(self.ranked_features_dict,
                                                                key=self.ranked_features_dict.get, reverse=True))
                                   if self.ranked_features_dict[i] > threshold))

    def get_features_std(self):
        try:
            if hasattr(self.classifier, 'feature_importances_'):
                self.std = np.std([self.classifier.feature_importances_], axis=0)
            elif hasattr(self.classifier, 'base_estimator'):
                self.classifier.fit(self.features, self.target)
                if hasattr(self.classifier.base_estimator, 'feature_importances_'):
                    self.std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_], axis=0)
                else:
                    self.std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_], axis=0)
            else:
                self.std = np.mean([tree.feature_importances_ for tree in self.classifier.estimators_], axis=0)

        except Exception, error:
            sys.stderr.write('\nGot error: %s. Features will be plotted without std.\n' % error)


    def plot_most_informative_features(self, filename, how_many=110, threshold=0.0):
            self.get_features_std()
            self.rank_features(how_many=how_many)
            self.show_most_informative_features(threshold=threshold)

            if self.std is not None: yerr = self.std[self.indices][0:how_many]
            else: yerr = None

            fig = plt.figure(figsize=(20, 30), dpi=100)
            plt.rc("font", size=20)
            plt.title("Feature importances")
            plt.bar(range(how_many), self.feature_importances_[self.indices][0:how_many],
                    yerr=yerr, color="g", align="center")
            plt.xticks(range(-1, how_many), [self.feat_names[n]
                                           for n in self.indices[0:how_many]], rotation=30, fontsize=6)
            plt.xlim([-1, how_many])
            plt.savefig('%s.png' % filename, format='png')
            # plt.show()

    def plot_tree_regression(self):

        clf_1 = tree.DecisionTreeRegressor(max_depth=2)
        clf_2 = tree.DecisionTreeRegressor(max_depth=5)
        clf_1.fit(self.features, self.target)
        clf_2.fit(self.features, self.target)

        y_1 = clf_1.predict(self.features)
        y_2 = clf_2.predict(self.features)


        plt.figure()
        plt.scatter(self.features, self.target, c="k", label="data")
        plt.plot(self.features, y_1, c="g", label="max_depth=2", linewidth=2)
        plt.plot(self.features, y_2, c="r", label="max_depth=5", linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Decision Tree Regression")
        plt.legend()
        plt.show()



    def predict(self, samples):
        self.predictions = [p for p in self.classifier.predict(samples)]

    def predict_probs(self, samples):
        self.predicted_probs = [[index + 1, x[1]]
                                for index, x in enumerate(self.classifier.predict_proba(samples))]

    def evaluate_model(self, test_target):
        self.classification_report = classification_report(test_target, self.predictions)
        self.confusion_matrix = confusion_matrix(test_target, self.predictions)
        sys.stdout.write('%s\n' % self.classification_report)
        sys.stdout.write('%s\n' % self.confusion_matrix)

    def evaluate_model_cv(self, folds=5):
        self.false_precision_scores = cross_validation.cross_val_score(
            self.classifier, self.features, self.target, cv=folds, scoring=self.false_precision_scorer)
        self.true_precision_scores = cross_validation.cross_val_score(
            self.classifier, self.features, self.target, cv=folds, scoring=self.true_precision_scorer)
        self.false_recall_scores = cross_validation.cross_val_score(
            self.classifier, self.features, self.target, cv=folds, scoring=self.false_recall_scorer)
        self.true_recall_scores = cross_validation.cross_val_score(
            self.classifier, self.features, self.target, cv=folds, scoring=self.true_recall_scorer)
        sys.stdout.write('\t\tPrecision\t\t\tRecall\nFalse\t%s (+/- %0.2f)\t%s (+/- %0.2f)\t\nTrue\t%s (+/- %0.2f)\t%s (+/- %0.2f)\n'
                         % (round(np.mean(self.false_precision_scores), 3), self.false_precision_scores.std(),
                            round(np.mean(self.false_recall_scores), 3), self.false_recall_scores.std(),
                            round(np.mean(self.true_precision_scores), 3), self.true_precision_scores.std(),
                            round(np.mean(self.true_recall_scores), 3), self.true_recall_scores.std()))

    def recursive_feature_elimination(self, scoring='precision', estimator=None):
        try:
            self.rfecv = RFECV(estimator=estimator, step=1, cv=3, scoring=scoring)
            self.rfecv.fit(self.features, self.target)
            sys.stdout.write("\nOptimal number of features : %d" % self.rfecv.n_features_)
            sys.stdout.write("\nRFECV score on optimal number of features: %s\n"
                             % self.rfecv.score(self.features, self.target))
        except Exception, error:
            sys.stderr.write('\nGot error: %s. Choose another classifier.\n' % error)


    def get_rfecv_chosen_feature_names(self):
        try:
            self.rfecv_chosen_feature_names = dict([(self.feat_names[i], self.rfecv.ranking_[i])
                                                    for i in self.rfecv.ranking_])
        except Exception, error:
            sys.stderr.write('\nGot error: %s. You have to call recursive_feature_elimination method first.\n' % error)
            sys.exc_clear()

    def show_rfecv_chosen_feature_names(self):
        self.get_rfecv_chosen_feature_names()
        sys.stdout.write('№\tFeature\t\t\tRank\n')
        sys.stdout.write('\n'.join('%s\t%s\t\t%s' % (n, i, self.rfecv_chosen_feature_names[i])
                                   for n, i in enumerate(sorted(self.rfecv_chosen_feature_names,
                                                                key=self.rfecv_chosen_feature_names.get))))

    def save_rfecv_plot(self, filename):
        try:
            fig = plt.figure(figsize=(20, 20), dpi=100)
            plt.rc("font", size=20)
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (nb of correct classifications)")
            plt.plot(range(1, len(self.rfecv.grid_scores_) + 1), self.rfecv.grid_scores_, linewidth=4.0, color='green')
            plt.grid()
            plt.savefig('%s.png' % filename, format='png')
            # plt.show()
        except Exception, error:
            sys.stderr.write('\nGot error: %s. You have to call recursive_feature_elimination method first.' % error)

    def bootstrap(self):
        self.bootstraper = classifiers['bootstrap']
        for train_index, test_index in self.bootstraper:
            self.train_feats = [self.features[i] for i in train_index]
            self.test_feats = [self.features[i] for i in test_index]
            self.train_target = [self.target[i] for i in train_index]
            self.test_target = [self.target[i] for i in test_index]
            self.classifier.fit(self.train_feats, self.train_target)
            self.predict(self.test_feats)
            self.evaluate_model(self.test_target)

    def get_bootstrap_informative_features(self):
        self.rank_features()
        self.cur_informative_features = self.ranked_features_dict
        for i in self.cur_informative_features:
            if i not in self.bootstrap_informative_features:
                self.bootstrap_informative_features[i] = list()
                self.bootstrap_informative_features[i].append(self.cur_informative_features[i])
            else:
                self.bootstrap_informative_features[i].append(self.cur_informative_features[i])

    def show_bootstrap_informative_features(self):
        try:
            sys.stdout.write('Bootstrapped informative features:\n')
            sys.stdout.write('\n'.join('%s.\t%s:\t%s' % (n, i, np.sum(self.bootstrap_informative_features[i]))
                             for n, i in enumerate(sorted(self.bootstrap_informative_features,
                                                          key=lambda x:
                                                          np.sum(self.bootstrap_informative_features[x]),
                                                          reverse=True))))
        except Exception, error:
            sys.stderr.write('Got error: %s. You have to call "get_bootstrap_informative_features" method first.\n'
                             % error)

    def save_model(self, fn):
        with open(fn, 'wb') as outfile:
            cPickle.dump(self.classifier, outfile)
        with open(fn + '.features', 'wb') as outfile:
            cPickle.dump(self.feat_names, outfile)

    def load_model(self, model, feat_names):
        with open(model, 'rb') as infile:
            self.classifier = cPickle.load(infile)
        with open(feat_names, 'rb') as infile:
            self.feat_names = cPickle.load(infile)



def run_rfecv(de_finder):
        scoring = de_finder.false_precision_scorer
        de_finder.recursive_feature_elimination(scoring=scoring, estimator=classifiers[classifier])
        de_finder.save_rfecv_plot('%s_RFECV_%s_cv_5_10estimators' % (classifier, str(scoring)))
        sys.stdout.write('False Precision\nBest chosen features: \n')
        de_finder.show_rfecv_chosen_feature_names()

        scoring = de_finder.false_recall_scorer
        de_finder.recursive_feature_elimination(scoring=scoring, estimator=classifiers[classifier])
        de_finder.save_rfecv_plot('%s_RFECV_%s_cv_3_10estimators' % (classifier, str(scoring)))
        sys.stdout.write('False Recall\nBest chosen features: \n')
        de_finder.show_rfecv_chosen_feature_names()
        sys.stdout.write('\nAll done!... at %s \n' % strftime("%a, %d %b %Y %H:%M:%S\n", gmtime()))


def _runall(fn, classifier, n_estimators=n_estimators):
        de_finder = DEFinder(fn)
        sys.stdout.write('Training model... at %s\n' % strftime("%a, %d %b %Y %H:%M:%S\n", gmtime()))

        de_finder.train_model(classifier=classifier)
        sys.stdout.write('Evaluating model %s... at %s\n' % (classifier, strftime("%a, %d %b %Y %H:%M:%S\n", gmtime())))
        de_finder.evaluate_model_cv(folds=3)

        de_finder.plot_most_informative_features('%s_most_inform_feats_std_gs_markup_cv_5_%s_estimators' %
                                                 (classifier, n_estimators), how_many=50, threshold=0.0)


def _profile_it(fn, classifier, n_estimators):
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput()):
        _runall(fn, classifier, n_estimators)


if __name__ == '__main__':
    in_fn = sys.argv[1]
    classifier = sys.argv[2]
    _runall(in_fn, classifier=classifier, n_estimators=n_estimators)




