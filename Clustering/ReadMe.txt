Spectral clustering worked well for this data set but it cannot add new data for clustering. 
Spectral Neural Net learns the graph laplacian transform, so new data can be added

The spectral net python environment was quite tricky to build and get compatible with many normal python libraries.
So this method requires passing data between the environments and organizing the scaler. Awkward.


Primary Env for helix fit:

#Goal cluster initial Dataset

#transform the helix fits into a format more useful for clustering, (midpoint distance map and dihedral angles)
#save in clustering data folder
python FitTransform.py data/Fits_4H.csv Clustering/data/dist_dihe_4H -c

#move to Clustering Directory
#cluster reference dataset
python Clustering.py -i dist_dihe_4H.npz -n refClus_new -d data/ -a data/ -c

#ensure -o directory exists
#generate dataset, from HelicalGenerator directory, generate 27877 to match original dataset size
python GenerateEndpoints.py -b 27877 -o Clustering/testData/gen_27877 -d

#move to Clustering Directory
#prepare generated dataset for clustering by spectral net based on the reference dataset clustering
#saved in data as to_predict.npz for loading by Clustering_SpectralNeuralNet.py
python Clustering.py -i gen_27877.npz -n refClus_new -d data -a data -e

#test removeClus
python Clustering.py -i removeClus7.npz -n refClus_rem7 -d testData -a data -e



SpectralNet environment:

#train spectral net
python Clustering_SpectralNeuralNet.py -i data/refClus_new.npz -n data/sNet -t

#load data prepared by Clustering.py and cluster with spectral net, using default specNet
python Clustering_SpectralNeuralNet.py -p -i data/refClus_new.npz -n data/sNet


Primary Env:

#compare clusters from the reference to the generated
python Clustering.py -l -n refClus_new- -d data -a data


