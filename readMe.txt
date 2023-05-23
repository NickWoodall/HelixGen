There are 3 modules here.

1. This top level directory fits and generates helical backbones.
	a. Fits the helical proteins with HelixFit.py 
    b. Transforms the parameters for mL with FitTransform.py
    c. Trains the GAN with TrainGAN.py with the endpoint distance map output from FitTransform.py.
    d. GenerateEndpoints.py uses the GAN to generate helical endpoints. (aka straight unlooped helices)
    e. LoopEndpoints.py finds many looped solutions around the general topology from the unlooped helices from the GAN.

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

	e. Clustering Directory is required to re-train the graphgen network with straight helices only


------------------------Primary Environment for Helix Fitting, GAN, Looping GAN Outputs---------------------------
#optimized for windows, need alter cuda/cudnn, tensorflow verions in linux
#compatibility issues for environments with pymol and tensorflow, pymol is used to read pdbs an embedded mistake at this point
#run commands in order to generate environment, yml files provided for thoroughness, but will not work

	conda create --name hgen python=3.8
	conda activate hgen
	conda install -c conda-forge cudatoolkit=10.1.243 cudnn=7.6.5
	pip install tensorflow-gpu==2.3
	conda install -c schrodinger pymol=2.4
	conda install shapely --channel conda-forge
	conda install pandas
	conda install numba
	conda install scikit-learn=0.23.2
	conda install matplotlib

	pip install lmfit
	pip install localization
	
	





-----------------------Workflow for generating and designing 4 helix bundles-------------------
#remakes csv object loop data without re-fitting the loops, use -r to redo the loop fits
python util_LoopCreation.py -j 

#Generate and loop 8 generated 4h topologies, must have recreated objects with python util_LoopCreation.py -j #
#defaults to BestGenerator in data/
python LoopEndpoints.py -b 8

#predict sequence and relax looped structures from above, move to GraphGen Directory, need to switch to pytorch/Rosetta environment
#move back to hgen environment, for other commands
python Predict_Design.py -i  ../output -n test1


#generate a single helical protein (more than four helices) along a guide set of endpoints 
#Defaults to the quarter circle line guide from the paper. To input your own start at the origin with the points with the first move 
#in the positive Z
# other options see HELP

python Guided_Midpoint_Buttress.py

-----------------------------------------Data sets available for retraining-------------------

-Retraining the GAN from scratch. Can use provided Fits_4H.csv and skip

4 helix reference pdb files into data/4H_dataset/models from [https://drive.google.com/drive/folders/1p8PgjTZsJ6Di8gbvVLx3lhUsygurGEYN?usp=sharing]


-for Retraining GraphGen_4H

4 helix reference models [above]
4 helix straight models into data/4H_dataset/ from [https://drive.google.com/drive/folders/1D7IA2jr1dIJFqSlbz30TDCbmJ229Myqc?usp=sharing], can be remade from reference

-------------------------Retraining Network From Scratch--------------------------------

#Fitting the dataset, you may skip this step and use the Fits_4H.csv provided
#A few warnings especially at the beginning are exptected, will take a few hours
python HelixFit.py data/4H_dataset/models data/Fits_4H_new

#convert fitted parameters to distance map and phi values of helical endpoints
#prints shape of saved numpy array
python FitTransform.py data/Fits_4H.csv data/Fits_4H_dm_phi -d

#train GAN 
python TrainGAN.py data/Fits_4H_dm_phi.npz -o FullSet


--------------------------------Training the GAN-------------------------------------


#copy the trained GAN and the min max scaler to the data folder for convenience

#You can train on a reduced version of the dataset using the -p option. This trains on 278 examples.
#Upping the epochs to 1000 for 278 examples.
python TrainGAN.py data/Fits_4H_dm_phi.npz -o OnePer -p 1 -e 1000

#transfer FullSet/ from retrain minmax scaler from log directory to data for this line to work
#produce generated loop structures from generator, use -a to not output the .pdb files and just get stats
python LoopEndpoints.py -i data/FullSet -a

-------------------Training GraphGen network to predict sequences------------------------

2. The Clustering directory using clusters the helical fits using a transformed version of the fits. The clusters are used
   to confirm the diversity of the GAN ouputs and divide the original dataset into train/test. To just prepare the test train
   datasets, move to Clustering/data/csv_saves/  otherwise follow the readme in the Clustering directory to remake.


#remake refclus data from csv in Clustering/data/csv_saves/
python Remake_Cluster_Scaler.py

#move the output to the upper data directory to Clustering/data/ (refData.npz and refData_scaler.gz)

You will need to make a new environment to support pytorch that supports pyrosetta (linux required for pyrosetta) training the network does not require pyrosetta
Activate this environment and move to the GraphGenDirectory

3. Train the GraphGen Network
##!!--Pyrosetta has been muted via its init method at the beginning of scripts which may not give error messages in the case of problems


#straighten the original dataset using helical fits via straighmin.py using pyrosetta
#requires pyrosetta, you may also download a straightened dataset from google drive (link above) instead
#make a directory str_models inside data/4H_dataset/
#move to Graphgen directory
python straightMin.py -i ../data/4H_dataset/models -o ../data/4H_dataset/str_models/


#prepare straightened helix data/ sequence for graphgen training
python PDBDataset.py -p 3 -d ../data/4H_dataset/models -s ../data/4H_dataset/str_models/ -o data/refSet -c ../Clustering/data/refData.npz

#train Graphgen
python TrainGraphGen.py -i data/refSet

#automatically generates in log directory time-stamped checkpoints and best checkpoint
#move best checkpoint to data for convienience

#test GraphGen network and generate, using 100 examples from the test 
python Test_GraphGen.py -s ../data/4H_dataset/models -i ../data/4H_dataset/str_models/ -g data/{checkpoint name here} -t -l 100



Libraries included in the repository

npose protein utilies for fragment assembly, backbone metrics, etc from Brian Coventry - many thanks!

#
Generative Models for Graph-Based Protein Design by John Ingraham, Vikas Garg, Regina Barzilay and Tommi Jaakkola, NeurIPS 2019.
https://github.com/jingraham/neurips19-graph-protein-design

# SpectralNet
![cc](https://user-images.githubusercontent.com/9156971/34493923-1abbabe8-efbc-11e7-8788-66c62bc91f4d.png)
