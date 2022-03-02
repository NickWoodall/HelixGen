import time, gzip
import time, os, sys, glob

import numpy as np

# Library code, need for imports in struct2seq
sys.path.insert(0,'struct2seq/')
sys.path.insert(0,f'experiments/')
from struct2seq import *
from utils import *
import data
import noam_opt

import util.AA_Exchange as aa

from pymol import cmd, stored, selector
from weakref import WeakKeyDictionary
import os
import argparse
import pickle

class StructureDataset_PDBDirec():
    """Replaces StructureDataset load from jsonl file with pdb load from direc """
    
    def __init__(self,dData,dStraight=None,limit=1,clusterLoad=None, test = False,err_msg=False):
        
        if dStraight is not None:
            #Get backbone features from dStraight and sequence from dData
            self.dStraight = dStraight 
            self.fileList_straight = os.listdir(self.dStraight)
            print(f'size of directory straight is {len(self.fileList_straight)}')
            self.dData = dData
            twoPart = True
        else: 
            twoPart = False
            self.dData = dData
            
                
        self.use_clus = False
        self.fileList = os.listdir(self.dData)
        self.data = []
        self.limit = limit #max load from directory
        self.nameList = []
        counter = 0
        
        self.data = [] #main info holder
        
        print(f'size of sequence directory is {len(self.fileList)}')

        
        if clusterLoad is not None:
            #hard coded from manually expecting clusters organized by AlignCluster
            #describes whether termini are together or apart, 
            #Approximately 15122 in together, 11556 in apart
            #seperate train and test class
            together=[0,1,2,8,9,14,15,17,18,19,20,23]
            apart = [3,4,5,6,7,10,11,12,13,16,24,25]
            
            if test:
                clusNums = apart
            else:
                clusNums = together
            
            self.use_clus = True
            
            
            with np.load(f'{clusterLoad}', allow_pickle=True) as rr:
                self.y_, self.y_train, self.X_train, self.featNames = [rr[f] for f in rr.files]
            self.cluster_labels = np.unique(self.y_)
            self.n_clusters = self.cluster_labels.shape[0]
            
            for x in clusNums:
                self.nameList.extend(self.y_train[np.where(self.y_==x)])
                
        #for test load of 
        if self.use_clus and not twoPart:
            intBack = 8 # get rids of _str in name for testGraphGen
        else:
            intBack = 4
            
        for x in self.fileList:

            if not x.endswith('.pdb'):
                continue
                
            if self.use_clus and x[:-intBack] in self.nameList:
                if twoPart:
                    if f'{x[:-4]}_str.pdb' in self.fileList_straight:
                        self.data.append(self.getFeatures_twoPart(x[:-4]))
                        counter+=1
                    else:
                        if err_msg:
                            print(f'{x} did not correspond to anything in the straight directory. Skipping.')
                else:
                    self.data.append(self.getFeatures(x[:-4]))
                    counter+=1
            elif not self.use_clus:
                if twoPart:
                    self.data.append(self.getFeatures_twoPart(x[:-4]))
                    counter+=1
                else:
                    self.data.append(self.getFeatures(x[:-4]))
                    counter+=1
            else:
                if err_msg:
                    print(f'{x} did not correspond to anything in the cluster reference. Skipping.')
                    
            if counter==self.limit:
                break
            
                
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def getFeatures_twoPart(self,name):
        """Loads in pdb via pymol. Extracts coordinates and sequence"""
        #clear pymol environment then load protein
        cmd.delete("all")
        cmd.load(f'{self.dData}/{name}.pdb')
        
        #get sequence from the original file
        stored.resn = []
        cmd.iterate_state(1, selector.process(f"{name} and n. CA"), "stored.resn.append(resn)")
        seq = ''
        for x in stored.resn:
            seq = f'{seq}{aa.aaCodeExchange(x)}'
            
        cmd.delete("all")
        
        ##Get Backbone data
        cmd.load(f'{self.dStraight}/{name}_str.pdb')
            
        #extract coordinates
        xyzN = cmd.get_coords(f"{name} and n. N", 1)
        xyzCA = cmd.get_coords(f"{name} and n. CA", 1)
        xyzC = cmd.get_coords(f"{name} and n. C", 1)
        xyzO = cmd.get_coords(f"{name} and n. O", 1)
        
        #save in dictionary
        entry= {'seq': seq,'coords': {'N':xyzN, 'CA':xyzCA, 'C':xyzC, 'O':xyzO}, 'num_chains':1, 'name': name, 'CATH':[''] }
        
        # Convert raw coords to np arrays
        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)
            
        return entry
    
    
    def getFeatures(self,name):
        """Loads in pdb via pymol. Extracts coordinates and sequence"""
        #clear pymol environment then load protein
        cmd.delete("all")
        cmd.load(f'{self.dData}/{name}.pdb')
        
        
        #get sequence from the original file
        stored.resn = []
        cmd.iterate_state(1, selector.process(f"{name} and n. CA"), "stored.resn.append(resn)")
        seq = ''
        for x in stored.resn:
            seq = f'{seq}{aa.aaCodeExchange(x)}'
            
        #extract coordinates
        xyzN = cmd.get_coords(f"{name} and n. N", 1)
        xyzCA = cmd.get_coords(f"{name} and n. CA", 1)
        xyzC = cmd.get_coords(f"{name} and n. C", 1)
        xyzO = cmd.get_coords(f"{name} and n. O", 1)
        
        #save in dictionary
        entry= {'seq': seq,'coords': {'N':xyzN, 'CA':xyzCA, 'C':xyzC, 'O':xyzO}, 'num_chains':1, 'name': name, 'CATH':[''] }
        
        # Convert raw coords to np arrays
        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)
            
        return entry
    
    
    
class StructureDataset_PDBFile():
    """Replaces StructureDataset load from jsonl file with pdb load from fileList """
    
    def __init__(self,fDataList,direcName=''):
        
        self.fData = fDataList
            
        self.data = []
        self.nameList = []
        
        self.data = [] #main info holder
        
        self.direcName = direcName
        
        for x in range(len(self.fData)):
            self.data.append(self.getFeatures(self.fData[x][:-4]))

            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def getFeatures(self,name):
        """Loads in pdb via pymol. Extracts coordinates and sequence"""
        #clear pymol environment then load protein
        cmd.delete("all")
        cmd.load(f'{self.direcName}/{name}.pdb')
        
        #get sequence from the original file
        stored.resn = []
        cmd.iterate_state(1, selector.process(f"{name} and n. CA"), "stored.resn.append(resn)")
        seq = ''
        for x in stored.resn:
            seq = f'{seq}{aa.aaCodeExchange(x)}'
            
        #extract coordinates
        xyzN = cmd.get_coords(f"{name} and n. N", 1)
        xyzCA = cmd.get_coords(f"{name} and n. CA", 1)
        xyzC = cmd.get_coords(f"{name} and n. C", 1)
        xyzO = cmd.get_coords(f"{name} and n. O", 1)
        
        #save in dictionary
        entry= {'seq': seq,'coords': {'N':xyzN, 'CA':xyzCA, 'C':xyzC, 'O':xyzO}, 'num_chains':1, 'name': name, 'CATH':[''] }
        
        # Convert raw coords to np arrays
        for key, val in entry['coords'].items():
            entry['coords'][key] = np.asarray(val)
            
        return entry
    
    
    
    







if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parseInput", help="[Required] 1: single pdb input direc 2: straight data 3: clusters and straight data", type=int,default=-1)
    parser.add_argument("-d", "--data", help="direc for pdb input with sequence (and structure if no -s)", default="../data/bCov_4H_dataset/BCov_Models")
    parser.add_argument("-s", "--straight", help="direc for pdb input for straightened structure.", default="../data/bCov_4H_dataset/BCov_Models_Straight" )
    parser.add_argument("-o", "--outName", help="Name for output.", default='data/refSet')
    parser.add_argument("-c", "--clusterName", help="Cluster File name", default="../Clustering/data/refData.npz")
    

    args = parser.parse_args()

    trainSet = None
    testSet = None
    
    if args.parseInput == 3:
        trainSet = StructureDataset_PDBDirec(args.data,dStraight=args.straight,limit=100000,
                                             clusterLoad=args.clusterName,test=False)
        testSet = StructureDataset_PDBDirec(args.data, dStraight=args.straight,limit=100000,
                                            clusterLoad=args.clusterName,test=True)
    elif args.parseInput == 2:
        trainSet = StructureDataset_PDBDirec(args.data,dStraight=args.straight,limit=100000)
    elif args.parseInput == 1:
        trainSet = StructureDataset_PDBDirec(args.data,limit=100000)
    
    if trainSet is not None:
        print(len(trainSet))
        with open(f'{args.outName}_train.pkl',"wb") as f:
            pickle.dump(trainSet,f)

    if testSet is not None:
        print(len(testSet))
        with open(f'{args.outName}_test.pkl',"wb") as f:
            pickle.dump(testSet,f)
            

        


        

# dData="../data/bCov_4H_dataset/BCov_Models/"
# dStraight = "../data/bCov_4H_dataset/BCov_Models_Straight/"
# dClus = "../Clustering/data/refData"
# trainSet = StructureDataset_PDBDirec(dData,dStraight=dStraight,limit=100000,clusterLoad=dClus,test=False)
# testSet = StructureDataset_PDBDirec(dData,dStraight=dStraight,limit=100000,clusterLoad=dClus,test=True)
# len(trainSet.data)
