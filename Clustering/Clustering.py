import os
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import seaborn as sns
import pickle

import sklearn.preprocessing as skp

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from sklearn.decomposition import KernelPCA, PCA

from sklearn.metrics import silhouette_samples
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering

from mpl_toolkits.mplot3d import Axes3D


from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,copysign,atan2
import math

import random
import datetime

import joblib
import subprocess
import argparse


#Process to Clusters
#1. Transform Fit of Helical Parameters to MidpointDist/ Dihedral Angles - FitTransform.py -c
#2. Apply StandardScaler Dist_Dihe [Save to use on new data later] - done with ClusterHelixParams
#3. Save cluster labels and features for training of Spectral Neural Net  - done with ClusterHelixParams
#4. Train Spectral Neural Net - with specNet environment
#5. Save weights for Siamese and Spectral Net - wtih specNet Environment

#Process New Data with Spectral Net
#1. Transform from Helical Parameters to MidpointDist/ Dihedral Angles FitTransform.py -c
#2. Load StandardScaler and apply to dist_dihe data - done with ClusterHelixParams
#3. Save scaled features for prediction by SpecNet - done with ClusterHelixParams
#4. Load SpecNet, fake train one epoch to intialize, Load weights -  with specNet environment
#5. Predict scaled dist_dihe data -  with specNet environment
#6. Save cluster predictions -  with specNet environment
#7. Load back with ClusterHelixParams

class ClusterHelixParams():
    
    def __init__(self,name,direc='data',scaler_direc='data', load_scaler=None,loadCluster = False):
        """Spectral Clustering, Visualization and connection to Spectral Neural Net"""
        #new cluster creates new scaler,  loads scaler from reference
        #ie training data = new cluster, mL generated data no new clusters
        
        if not loadCluster:
            self.name = name[:-4]#remove .npz
        else:
            self.name = name
        self.direc=f'{direc}/'
        scaler_direc=f'{scaler_direc}/'
        rr = np.load(f'{self.direc}{self.name}.npz', allow_pickle=True)
        
        
        if loadCluster:
            #Loading data for plotting
            self.y_, self.y_train, self.X_train, self.featNames = [rr[f] for f in rr.files]
            #self.y_, self.y_train, self.X_train = [rr[f] for f in rr.files]
            self.stdsc = joblib.load(f'{scaler_direc}{self.name}_scaler.gz')
            self.cluster_labels = np.unique(self.y_)
            self.n_clusters = self.cluster_labels.shape[0]
                 
        elif load_scaler is not None:
            #loading data to include in clustering via spectral net
            self.X_train, self.y_train, self.featNames = [rr[f] for f in rr.files]
            self.stdsc = joblib.load(f'{scaler_direc}{load_scaler}_scaler.gz')
            self.X_train = self.stdsc.transform(self.X_train)
        else:
            #Loading data for clustering
            self.X_train, self.y_train, self.featNames = [rr[f] for f in rr.files]
            self.stdsc = skp.StandardScaler()
            self.X_train = self.stdsc.fit_transform(self.X_train)
            
            
        self.featsD = self.X_train[:,:-8] #removed length /phi values
        
    
    def saveClusters(self,name,direc='data'):
        
        direc = f'{direc}/'
        np.savez_compressed(f'{direc}{name}.npz', y_ = self.y_ , y_train = self.y_train, 
                            X_train=self.X_train, featNames = self.featNames)
        joblib.dump(self.stdsc, f'{direc}{name}_scaler.gz')
        
    
    def save_for_spectralNet(self,direc='data/', name='to_predict'):
        
        np.savez_compressed(f'{direc}{name}.npz',data = self.featsD)
    
    def load_for_spectralNet(self,direc='data/',newClust='specNet_predicted',
                             origClust='specNet_predicted_original_clusters'):
        
        rr = np.load(f'{direc}{newClust}.npz', allow_pickle=True)
        
        self.y_ = [rr[f] for f in rr.files][0]
        
        self.cluster_labels = np.unique(self.y_)
        self.n_clusters = self.cluster_labels.shape[0]
        
        rr = np.load(f'{direc}{origClust}.npz', allow_pickle=True)
        
        self.original_dataset_y_ = [rr[f] for f in rr.files][0]
        
        return self.y_
        
    
    def sc(self,n_clusters=26,random_state=0, affinity='nearest_neighbors', n_neighbors=10):
        """Use spectral clustering to cluster."""
        
        self.sc = SpectralClustering(assign_labels='discretize',affinity=affinity, n_neighbors=n_neighbors,n_clusters=n_clusters,random_state=random_state)
        self.y_ = self.sc.fit_predict(self.featsD)
        
        self.cluster_labels = np.unique(self.y_)
        self.n_clusters = self.cluster_labels.shape[0]
        
        return self.y_

    
    def pcaPlot(self,plotNeg=True):
        
        pca = PCA(n_components=2)
        xt_pca = pca.fit_transform(self.featsD)
        
        for i, c in enumerate(self.cluster_labels):
            if c==-1 and not plotNeg:
                continue
            color = cm.jet(float(i+1)/self.n_clusters)
            axobj = plt.scatter(xt_pca[self.y_==c,0], xt_pca[self.y_==c,1],color = color, alpha = 0.5)
            
        return axobj
            
    def pca3DPlot(self,plotNeg=False):
        
        # Run The PCA
        pca = PCA(n_components=3)
        xt_pca=pca.fit_transform(self.featsD)
        
        
        
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')
        

        for i, c in enumerate(self.cluster_labels):
            if c==-1 and not plotNeg:
                continue
            color = cm.jet(float(i+1)/self.n_clusters)
            ax.scatter3D(xt_pca[self.y_==c,0], xt_pca[self.y_==c,1],xt_pca[self.y_==c,2] ,color = color, alpha = 0.5)                   
    
    
    def tSNE_plot(self, perplexity=30.0):
        """Make tSNE plot"""
        
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        tn = TSNE(perplexity=perplexity)
        
        sneReduced=tn.fit_transform(self.featsD)
        
        
        palette = sns.color_palette("hls", self.n_clusters)
        sns.scatterplot(x=sneReduced[:,0], y=sneReduced[:,1], hue=self.y_, palette=palette)
    
    def silPlot(self):
        """Make Silhouette Plot"""

        self.silhouette_vals = silhouette_samples(self.featsD, self.y_, metric='euclidean')
        
        y_ax_lower, y_ax_upper = 0,0
        yticks=[]
        
        for i, c in enumerate(self.cluster_labels):
            c_silhouette_vals = self.silhouette_vals[self.y_ == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / self.n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper),
                    c_silhouette_vals,
                    height = 1.0,
                    edgecolor = 'none',
                    color=color)
            yticks.append((y_ax_lower+y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)
        silhouette_avg = np.mean(self.silhouette_vals)
        plt.axvline(silhouette_avg, color='red', linestyle="--")
        plt.yticks(yticks, self.cluster_labels +1)
        plt.xlabel('Silhouette Coeffecient')
        
    def compare_Clusters(self):
        """Plots cluster distribution from ref data set and generated helices.Same number of samples per cluster."""
        oClus = []
        nClus = []
        
        for x in range(self.n_clusters):
            oClus.append(len(np.where(self.original_dataset_y_==x)[0]))
            nClus.append(len(np.where(self.y_== x)[0]))
            
        nums =  list(range(self.n_clusters))
        nums.extend(nums)
        oldList = ['reference' for x in range(self.n_clusters)]
        newList = ['generated' for x in range(self.n_clusters)]
        oldList.extend(newList)
        oClus.extend(nClus)
        
        data1 = {'cluster':nums, 'amount': oClus, 'type': oldList}
        
        df = pd.DataFrame(data=data1)
        print(df)
        sns.barplot(x="cluster", y="amount", data=data1, hue='type')
        return df 
        
    def getRandomExamples(self,numRetrieve=1):
        """Check clustering by looking at random members of clusters"""
        cluster_examples = []
        
        for x in range(len(self.cluster_labels)):
            cluster = self.y_train[self.y_== self.cluster_labels[x]]
            cluster_examples.append(random.choice(cluster,k=numRetrieve))
             
        return cluster_examples
    
    def getSilSortedExamples(self,numRetrieve=1):
        """Check clustering by checking the most similar members of clusters"""
        self.silhouette_vals = silhouette_samples(self.featsD, self.y_, metric='euclidean')
        
        cluster_examples = []
        
        for i, c in enumerate(self.cluster_labels):
            c_silhouette_vals = self.silhouette_vals[self.y_ == c]
            c_names = self.y_train[self.y_ == c]
            index = c_silhouette_vals.argsort()
            
            self.clustered_poses.append([])
            c_silhouette_vals = c_silhouette_vals[index]
            cluster_examples.append(c_names[index])
            
        return cluster_examples
                
def cluster_distant_dihedral_new(name,direc,outname,outdirec,n_clusters=26):
    """Uses spectral cluster to cluster by midpoints of helix distance and dihedrals of helices."""
    chp = ClusterHelixParams(name, direc=direc)
    chp.sc(n_clusters=n_clusters,random_state=0, affinity='nearest_neighbors', n_neighbors=10)
    #chp.pcaPlot() #Visualize!
    chp.saveClusters(outname,direc=outdirec)
    print("Clusters_Saved")
    
def prep_for_specNetCluster(name,scaler,inDirec='data',scaler_direc='data'):
    """Applies scaler and saves for loading by spectral net code."""
    chp = ClusterHelixParams(name,direc=inDirec,load_scaler=scaler,scaler_direc=scaler_direc) #scale dat with prevscaler
    chp.save_for_spectralNet()# save for loading by spectral net, produces to_predict.npz in data/

def load_and_compare_specNet_clusters(refClus_name,inDirec='data',scaler_direc='data'):
    """Load clusters from spectral net"""
    chp = ClusterHelixParams(refClus_name, direc=inDirec, scaler_direc=scaler_direc,loadCluster=True)
    chp.load_for_spectralNet()
    chp.compare_Clusters()
    return chp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster data with spectral clustering. Add new data to ref clustering with spectral neural net. See references for original spectral neural net.")
    
    parser.add_argument("-n", "--refClus_name", help="Name for reference clustering. Output for initial clustering. Input to add new data to cluster.")
    parser.add_argument("-i", "--input_data", help="Input distance dihedrals. Features for clustering.")
    parser.add_argument("-d", "--data_direc", help="Directory that contains input distance dihedrals feature data.")
    parser.add_argument("-s", "--scaler_name", help="Input name of the scaler.")
    parser.add_argument("-a", "--scaler_direc", help="directory that contains the scaler.")
                        
    parser.add_argument("-c","--cluster", help="Initial clustering of distance dihedral features. Requires -n, -i, -d, -a",action="store_true")
    parser.add_argument("-e","--prepSpecNet", help="Scale and save data for cluster prediction by Spectral Net.   Requires -n, -i, -d, -a",action="store_true")
    parser.add_argument("-l","--compareSpecClusters", help="Load and compare spectrally predicted clusters.  Requires -n, -i, -d, -a", action="store_true")
    args = parser.parse_args()                    
                        
    if args.cluster:
        #requires -n, -i, -d, -a
        if args.refClus_name and args.input_data and args.data_direc and args.scaler_direc:
            cluster_distant_dihedral_new(args.input_data, args.data_direc, args.refClus_name, args.scaler_direc)
        else:
            print("Requires additional flags. Check help.")
    elif args.prepSpecNet: 
        #requires -i, -d, -n -a
        if args.refClus_name and args.input_data and args.data_direc and args.scaler_direc:
            prep_for_specNetCluster(args.input_data,args.refClus_name,inDirec=args.data_direc,scaler_direc=args.scaler_direc)
        else:
            print("Requires additional flags. Check help.")
                        
    elif args.compareSpecClusters:
        #requires -d, -n -a
        if args.refClus_name and args.data_direc and args.scaler_direc:
            cc = load_and_compare_specNet_clusters(args.refClus_name,inDirec=args.data_direc, scaler_direc=args.scaler_direc)
        else:
            print("Requires additional flags. Check help.")
        
                        
    
    
# cluster_distant_dihedral('dd_test.npz', 'testData', 'test_clusterBcov', 'testData')
# prep_for_specNetCluster('gen_dd.npz','test_clusterBcov',inDirec='../test',scaler_direc='testData') 
# cc = load_specNet_clusters('gen_dd.npz','test_clusterBcov',inDirec='../test',scaler_direc='testData')

  


