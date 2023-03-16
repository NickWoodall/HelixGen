

-----------Environment torchGraph------------
#these recommended install for environment for incompatibilities with pymol/pytorch/ pyrosetta
#.yml file provided for thouroughness, but follow commands below

conda create -n hdes python=3.8
conda install -c schrodinger pymol=2.4
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch -c https://levinthal:paradox@conda.rosettacommons.org
conda install matplotlib
conda install numba
conda install seaborn
pip install scikit-learn
pip install lmfit


----------------Predict Sequence and then Mutate/Relax Structure with Pyrosetta----------------


#in GraphGen Directory, predict sequence and relax looped structures from above, need to switch to pytorch/Rosetta environment
python Predict_Design.py -i  ../output -n test1


---------------Re-train the GraphGen Network------------------------------

-for Retraining GraphGen_4H

4 helix reference models into data/4H_dataset/ from [https://drive.google.com/drive/folders/1p8PgjTZsJ6Di8gbvVLx3lhUsygurGEYN?usp=sharing]
4 helix straight models into data/4H_dataset/ from [https://drive.google.com/drive/folders/1D7IA2jr1dIJFqSlbz30TDCbmJ229Myqc?usp=sharing] can be remade from reference


#straighten the original dataset using helical fits via straighmin.py using pyrosetta
#move to Graphgen directory
python straightMin.py -i ../data/4H_dataset/models -o ../data/4H_dataset/str_models/


#prepare straightened helix data/ sequence for graphgen training
python PDBDataset.py -p 3 -d ../data/4H_dataset/models -s ../data/4H_dataset/str_models/ -o data/refSet -c ../Clustering/data/refData.npz

#train Graphgen
python TrainGraphGen.py -i data/refSet


#test GraphGen network and generate, using 100 examples from the test 
python Test_GraphGen.py -s ../data/4H_dataset/models -i ../data/4H_dataset/str_models/ -g data/{checkpoint name here} -t -l 100


