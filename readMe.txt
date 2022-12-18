*Deal with advice for environment creation, of fa_tfpy, specNetGPU, and TorchGraph
*check restricted google drive downloads
*check clustering specNet
*Check Clustering Flags


There are 3 modules here.

1. This top level directory fits and generates helical backbones.
	a. Fits the helical proteins with HelixFit.py . 
        b. Transforms the parameters for mL with FitTransform.py
        c. Trains the GAN with TrainGAN.py with the endpoint distance map output from FitTransform.py.
        d. GenerateEndpoints.py uses the GAN to generate helical endpoints. (aka straight unlooped helices)
        e. LoopEndpoints.py finds many looped solutions around the general topology from the unlooped helices from the GAN.
		a. Can generate endpoints from GAN as part of script

2. The GraphGen directory contains code to predict the sequence of 4 helix proteins and relax to a reasonable structure.
	a. Prepares PDB structures for prediction by GraphGen Network with PDBDataset.py
	b. PredictSeq.py predicts the sequence of pdb structure using the graphgen Network
	c. Predict_Design.py predicts the sequence with Graphgen, mutates then relaxes the structure using Rosetta.
		c1. RelaxFastdesgin.py contains the Rosetta Methods
	d. Test_GraphGen.py return sequence prediction statistics or rosetta score stats for a graphgen model.
		d1. TrainGraphGen.py trains the graphgen network. Requires d2/d3.
		d2. straightMin.py straightens the dataset for training Graphgen on straightened structure.
		d3. The reference clusters must be remade from Clustering/data/csv_saves/Remake_Cluster_Scaler.py
			to maintain produce train/test split

-----Required Datasets for looping endpoints:

download loop structures into data/ from [https://drive.google.com/file/d/16GuyJPYWOEW0Ud-sPrklGuFuuKwk6sjO/view?usp=sharing]

------Required Datasets for Retraining the GAN from scratch. Can use Fits_4H.csv:

4 helix reference models into data/4H_dataset/ from [https://drive.google.com/drive/folders/1p8PgjTZsJ6Di8gbvVLx3lhUsygurGEYN?usp=sharing] #restricted

------Datasets for Retraining GraphGen_4H

4 helix reference models into data/4H_dataset/ from [https://drive.google.com/drive/folders/1p8PgjTZsJ6Di8gbvVLx3lhUsygurGEYN?usp=sharing] #restricted
4 helix straight models into data/4H_dataset/ from [https://drive.google.com/drive/folders/1D7IA2jr1dIJFqSlbz30TDCbmJ229Myqc?usp=sharing] #restricted, can be remade from reference


-----------------------Workflow for generating and designing 4 helix bundles-------------------
#Generate and loop 8 generated 4h topologies
python LoopEndpoints.py -b 8

#in GraphGen Directory, predict sequence and relax looped structures from above, need to switch to pytorch/Rosetta environment
python Predict_Design.py -i  ../output -n test1

##!!--Pyrosetta has been muted via its init method in a lot of places which may not give error messages in the case of 
--------------------------------Training the GAN-------------------------------------

#Fitting the dataset, you may skip this step and use the Fits_4H.csv provided
# a few proteins in the dataset don't fit the minimum length requirements and will fail generating a warning, will take a few hours
python HelixFit.py data/4H_dataset/models data/Fits_4H_new

#convert fitted parameters to distance map and phi values of helical endpoints
#prints shape of saved numpy array
python FitTransform.py data/Fits_4H.csv data/Fits_4H_dm_phi -d

#train GAN and save the loss plots
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


#remake refclus data from csv in Clustering/data/csv_saves/
python Remake_Cluster_Scaler.py

#move the output to the upper data directory to Clustering/data/ (refData.npz and refData_scaler.gz)

You will need to make a new environment to support pytorch that supports pyrosetta (ubuntu required for pyrosetta) training the network does not require pyrosetta
Activate this environment and move to the GraphGenDirectory

3. Train the GraphGen Network

#straighten the original dataset using helical fits via straighmin.py using pyrosetta
#requires pyrosetta, you may also download a straightened dataset from google drive (link above) instead
#make a directory str_models inside data/4H_dataset/
#move to Graphgen direcotry
python straightMin.py -i ../data/4H_dataset/models -o ../data/4H_dataset/str_models/


#prepare straightened helix data/ sequence for graphgen training
python PDBDataset.py -p 3 -d ../data/4H_dataset/models -s ../data/4H_dataset/str_models/ -o data/refSet -c ../Clustering/data/refData.npz

#train Graphgen
python TrainGraphGen.py -i data/refSet

#automatically generates in log directory time-stamped checkpoints and best checkpoint
#move best checkpoint to data for convienience

#test GraphGen network and generate, using 100 examples from the test 
python Test_GraphGen.py -s ../data/4H_dataset/models -i ../data/4H_dataset/str_models/ -g data/{checkpoint name here} -t -l 100


