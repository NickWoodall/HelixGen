*Deal with advice for environment creation, of fa_tfpy, specNetGPU, and TorchGraph
*make sure loop csv and pdb dataset is included in github
*test loop remake with just functional


There are 3 modules here.

1. This top level directory fits and generates helical backbones.
	a. Fits the helical proteins with HelixFit.py . 
        b. Transforms the parameters for mL with FitTransform.py
        c. Trains the GAN with TrainGAN.py with the endpoint distance map output from FitTransform.py.
        d. GenerateEndpoints.py uses the GAN to generate helical endpoints. (aka straight unlooped helices)
        e. LoopEndpoints.py finds many looped solutions around the general topology from the unlooped helices from the GAN.


--------------------------------Training the GAN-------------------------------------

#Fitting the dataset
# a few proteins in the dataset don't fit the minimum length requirements and will fail generating a warning
python HelixFit.py [path/to/4Hdataset/direcotry] data/Fits_4H

#convert fitted parameters to distance map and phi values of helical endpoints
#prints shape of saved numpy array
python FitTransform.py data/Fits_4H.csv data/Fits_4H_dm_phi -d

python TrainGAN.py data/Fits_4H_dm_phi.npz -o FullSet -s

#copy the trained GAN and the min max scaler to the data folder for convenience

#if you used -s in TrainGAN.py you may plot and view the loss data in util/plot.py
python util/plot.py -i log/GAN_FullSet/loss/loss_FullSet.pkl -o log/GAN_FullSet/loss/lossplot.png

#You can train on a reduced version of the dataset using the -p option. This trains on 278 examples.
#Upping the epochs to 1000 for 278 examples.
python TrainGAN.py data/Fits_4H_dm_phi.npz -o OnePer -s -p 1 -e 1000

#remakes csv object loop data without re-fitting the loops, use -r to redo the loop fits
python util_LoopCreation.py -j

#produce generated loop structures from generator, use -a to not output the .pdb files and just get stats
python LoopEndpoints.py -i data/FullSet -a

-------------------Training GraphGen network to predict sequences------------------------

2. The Clustering directory using clusters the helical fits using a transformed version of the fits. The clusters are used
   to confirm the diversity of the GAN ouputs and divide the original dataset into train/test. To just prepare the test train
   datasets, move to Clustering/data/csv_saves/  otherwise follow the readme in the Clustering directory.


#remake refclus data from csv
python Remake_Cluster_Scaler.py

#move the output to the upper data directory (refData.npz and refData_scaler.gz)

You will need to make a new environment to support pytorch that supports pyrosetta (ubuntu)
And move to an environment that supports pytorch

3. Train the GraphGen Network

#straighten the original dataset using helical fits via straighmin.py using pyrosetta
#make a directory str_models inside data/4H_dgit puataset/
#move to 

python straightMin.py -i [path/to/4H_data] -o ../data/4H_dataset/str_models/


#prepare straightened helix data/ sequence for graphgen training
python PDBDataset.py -p 3 -d [path/to/4H_dataset (sequences)] -s [path/to/4H_dataset straightstruct] -o data/refSet -c ../Clustering/data/refData.npz

#train Graphgen
python TrainGraphGen.py -i data/refSet

#automatically generates in log directory time-stamped checkpoints and best checkpoint
#move best chekc

#test GraphGen network
python Test_GraphGen.py -i ../../BCov_Models -i ../data/4H_dataset/str_models -g log/sr2/best_checkpoint_epoch59.pt -t -l 100


-----------------------Workflow for generating and designing 4 helix bundles-------------------
#Generate and loop 8 generated 4h topologies
python LoopEndpoints.py -b 8

#in GraphGen Directory, predict sequence and relax looped structures from above
python Predict_Design.py -i  ../output -n test1