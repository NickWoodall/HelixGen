import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
import tensorflow as tf
from collections import defaultdict
import sys, os
# add directories in src/ to path. Using Spectral Net, see reference
sys.path.insert(0, 'SpectralNet-master/src/applications/')
sys.path.insert(0, 'SpectralNet-master/src/')
from spectralnet import run_net
from core.data import get_data

from sklearn.neighbors import LSHForest
import joblib

# '''
# spectralnet.py: contains run function for spectralnet
# '''
import sys, os, pickle
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score as nmi

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop

from core import train
from core import costs
from core import networks
from core.layer import stack_layers
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds
import argparse
#parameters for spectralNet

params = defaultdict(lambda: None)
#change to dset  = mnist and codespace = True
general_params = {
        'dset': 'new',                  # dataset: reuters / mnist
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        'siam_batch_size': 128,             # minibatch size for siamese net
        }
params.update(general_params)
#         'train_labeled_fraction':True,
#         'val_labeled_fraction':True,
my_params = {

        'n_clusters': 26,                   # number of clusters in data
        'use_code_space': False,             # enable / disable code space embedding
        'affinity': 'siamese',              # affinity type: siamese / knn
        'n_nbrs': 10,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
                                            # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
                                            # sampled from the datset

        'siam_k': 2,                        # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
                                            # a 'positive' pair by siamese net

        'siam_ne': 50,                     # number of training epochs for siamese net
        'spec_ne': 150,                     # number of training epochs for spectral net
        'siam_lr': 1e-3,                    # initial learning rate for siamese net
        'spec_lr': 1e-3,                    # initial learning rate for spectral net #hardcoded in network.py?
        'siam_patience': 10,                # early stopping patience for siamese net
        'spec_patience': 20,                # early stopping patience for spectral net
        'siam_drop': 0.1,                   # learning rate scheduler decay for siamese net
        'spec_drop': 0.1,                   # learning rate scheduler decay for spectral net
        'batch_size': 1024,                 # batch size for spectral net
        'siam_reg': None,                   # regularization parameter for siamese net
        'spec_reg': None,                   # regularization parameter for spectral net
        'siam_n': None,                     # subset of the dataset used to construct training pairs for siamese net
        'siamese_tot_pairs': 600000,        # total number of pairs for siamese net
        'arch': [                           # network architecture. if different architectures are desired for siamese net and
                                            #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 512},
            {'type': 'relu', 'size': 10},
            ],
        'use_approx': False,                # enable / disable approximate nearest neighbors
        'use_all_data': False,               # enable to use all data for training (no test set)
        }
params.update(my_params)



def spectralNet_FromWeights(data,params,siamWeightsPath,specWeightsPath):
    
     #
    # UNPACK DATA
    #

    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
    x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral']['train_unlabeled_and_labeled']
    x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

    if 'siamese' in params['affinity']:
        pairs_train, dist_train, pairs_val, dist_val = data['siamese']['train_and_test']

    x = np.concatenate((x_train, x_val, x_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)

    if len(x_train_labeled):
        y_train_labeled_onehot = OneHotEncoder().fit_transform(y_train_labeled.reshape(-1, 1)).toarray()
    else:
        y_train_labeled_onehot = np.empty((0, len(np.unique(y))))

    #
    # SET UP INPUTS
    #

    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    batch_sizes = {
            'Unlabeled': params['batch_size'],
            'Labeled': params['batch_size'],
            'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
            }

    input_shape = x.shape[1:]

    # spectralnet has three inputs -- they are defined here
    inputs = {
            'Unlabeled': Input(shape=input_shape,name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape,name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape,name='OrthonormInput'),
            }

    #
    # DEFINE AND TRAIN SIAMESE NET
    #

    # run only if we are using a siamese network
    if params['affinity'] == 'siamese':
        siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), y_true)

        history = siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
                params['siam_lr'], params['siam_drop'], params['siam_patience'],
                1, params['siam_batch_size'])
        siamese_net.net.load_weights(siamWeightsPath, by_name=True)

    else:
        siamese_net = None

    #
    # DEFINE AND TRAIN SPECTRALNET
    #

    spectral_net = networks.SpectralNet(inputs, params['arch'],
            params.get('spec_reg'), y_true, y_train_labeled_onehot,
            params['n_clusters'], params['affinity'], params['scale_nbr'],
            params['n_nbrs'], batch_sizes, siamese_net, x_train, len(x_train_labeled))

    spectral_net.train(
            x_train_unlabeled, x_train_labeled, x_val_unlabeled,
            params['spec_lr'], params['spec_drop'], params['spec_patience'],
            1)

    spectral_net.net.load_weights(specWeightsPath, by_name=True)

    print("finished training")

    #
    # EVALUATE
    #

    #get final embeddings
    x_spectralnet = spectral_net.predict(x)

    #get accuracy and nmi
    kmeans_assignments, km = get_cluster_sols(x_spectralnet, ClusterClass=KMeans, n_clusters=params['n_clusters'], init_args={'n_init':10})
    
    kmeans_assignments = km.predict(x_spectralnet)
    
    y_spectralnet, _ = get_y_preds(kmeans_assignments, y, params['n_clusters'])
    print_accuracy(kmeans_assignments, y, params['n_clusters'])

    return km, siamese_net, spectral_net #,x_spectralnet, y_spectralnet


def train_and_save(outName= 'data/testModel', dataIn='data/refData.npz'):
    #testData for training
    # tf.test.is_gpu_available()
    rr = np.load(dataIn, allow_pickle=True)
    #X_train data is the feature for spectral clustering(Midpoint distance + dihedral angles between helices)
    #16 total, eight each
    #y_name is the name of the protein
    #y_ is the assigned cluster labels from real spectral clustering
    y_start , y_name_start, X_train_start, featNames  = [rr[f] for f in rr.files]

    X_train_start = X_train_start[:,:-8] #remove phi values and length

    #Warning! test Train splits hard coded here for original data
    X_test = X_train_start[22000:,:]
    y_test = y_start[22000:]

    X_train = X_train_start[:22000,:]
    y_train = y_start[:22000]

    #run this to organize the data into the approriate dictionary formats
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    new_dataset_data = (X_train, X_test, y_train, y_test)
    ata = get_data(params,new_dataset_data)
    
    #train net with this code
    siamese_net_model, spectral_net_model, x_spectralnet, y_spectralnet,km = run_net(ata, params)
    
    #save the weights for loading by spectralNet_FromWeights
    spectral_net_model.net.save_weights(f'{outName}_spectral_net.tf')
    siamese_net_model.net.save_weights(f'{outName}_siamese_net.tf')


def load_and_predict(out_name='specNet_predicted',refData='data/refData.npz', specWeightsPath='', siamWeightsPath=''):
    #testData used in the training, needs to be reloaded for prediction of clusters,for re-predicting
    #original clusters to maintain consistent cluster numbers
    #Warning! test Train splits hard coded here for original data
    
    # tf.test.is_gpu_available()
    rr = np.load(refData, allow_pickle=True)
    #X_train data is the feature for spectral clustering(Midpoint distance + dihedral angles between helices)
    #16 total, eight each
    #y_name is the name of the protein
    #y_ is the assigned cluster labels from real spectral clustering
    y_start , y_name_start, X_train_start, featNames  = [rr[f] for f in rr.files]

    X_train_start = X_train_start[:,:-8] #remove phi values and length


    X_test = X_train_start[22000:,:]
    y_test = y_start[22000:]

    X_train = X_train_start[:22000,:]
    y_train = y_start[:22000]

    #run this to organize the data into the approriate dictionary formats
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    new_dataset_data = (X_train, X_test, y_train, y_test)
    ata = get_data(params,new_dataset_data)
    #due to the special layers and old format, remake the network per the regular training code
    #and then load the weights after one epoch of training
    km, siamese_net_model, spectral_net_model = spectralNet_FromWeights(ata, params,siamWeightsPath,
                                                                       specWeightsPath)
    
    #load data to predict
    direc = 'data/'
    name = 'to_predict'
    rr = np.load(f'{direc}{name}.npz', allow_pickle=True)
    data = [rr[f] for f in rr.files]
    
    #predict and assign clusters for new data
    x_spec = spectral_net_model.predict(data[0])
    clusters_assignments = km.predict(x_spec)
    
    #save the data to give back to clustering class (new python environment easier to use)
    np.savez_compressed(f'{direc}{out_name}.npz',data = clusters_assignments)
    
    #repredict the inital data to get consistent cluster numbers 
    #original dataset assignments

    x_spec_orig = spectral_net_model.predict(X_train_start)
    cluster_assignments_orig = km.predict(x_spec_orig)

    np.savez_compressed(f'{direc}{out_name}_original_clusters.npz', data=cluster_assignments_orig)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Clusters Using Spectral Net. Requires a different environment")
    parser.add_argument("-i", "--input_data", help="Input Data to train clustering. the Reference data for clustering. From ClusterHelixParams", default='data/refData.npz')
    parser.add_argument("-n", "--net_name", help="Name for Spectral Net Weights saves")
    parser.add_argument("-t","--train", help="train spectral net. Requires -i,-n", action="store_true")
    parser.add_argument("-p","--predict_clusters", help="Use spectral neural net to predict the clustering of the new data. Requires -o. Optionally specify the -m,-i,-s.", action="store_true")
    

    args = parser.parse_args()
    
    if args.train:
        train_and_save(outName= args.net_name, dataIn=args.input_data)
    elif args.predict_clusters:
        load_and_predict(refData=args.input_data,siamWeightsPath=f'{args.net_name}_siamese_net.tf',
                specWeightsPath=f'{args.net_name}_spectral_net.tf')
        print('Data saved')
        
    
    