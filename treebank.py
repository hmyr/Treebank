"""

"""
import os
import sys
from model import DEFinder, RandomForestClassifierWithCoef
import dep_features
from clustering import DECluster
import logging


def process_dir(dir, out_feats_dir, outfn):
    for fn in os.listdir(dir):
        logging.info('Extracting features from file {}')
        fpath = os.path.join(dir, fn)
        docname='_features.csv' % fn
        dep_features.get_features(infile=fpath, featsdir=out_feats_dir, docname=docname, param='head')
        extr_feats_fn = os.path.join(out_feats_dir, docname)
        logging.info('Predicting...')
        definder.target, definder.features, definder.feat_names, classifier.ids = \
            DECluster.read_features(extr_feats_fn, target_feat_name='childCheck', delimiter=',', id_name='id')
        definder.predict(definder.features)
        logging.info('Saving predictions...')
        definder.map_predicted_to_ids(what=False)
        definder.save_predictions(outfn)




if __name__ == '__main__':
    markup_dir = sys.argv[1]
    out_feats_dir = sys.argv[2]
    out_fn = sys.argv[3]
    modelname = sys.argv[4]
    definder = DEFinder()
    definder.load_model(modelname, modelname + '.features')
    # process_dir(markup_dir, out_feats_dir, out_fn)

