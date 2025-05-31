import pickle
import sys

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp

# I removed most of the process of filtering labels, since it's not our goal

def load_data(prefix='DBLP'):
    from HINormer.utils.data_loader import data_loader
    dl = data_loader('laion_clap/HINormer/data/'+prefix)
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.8
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    #print(train_idx)
    # np.random.shuffle(train_idx) sorry we can't shuffle our data
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[split:]
    train_idx = train_idx[:split]
    #print(train_idx)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB' and prefix != 'IMDB-HGB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}

    # let's reassign the train, val, test value for my dataset
    train_nodes_total = dl.nodes['count'][0]

    train_idx = 0.8*train_nodes_total
    val_idx = train_idx + 0.2*train_nodes_total
    test_idx = 1

    train_val_test_idx['train'] = int(train_idx)
    train_val_test_idx['val'] = int(val_idx)
    train_val_test_idx['test'] = int(test_idx)
    return features,\
           adjM, \
           labels,\
           train_val_test_idx,\
            dl
