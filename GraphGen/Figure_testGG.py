import sys


import PredictSeq as ps
import time
import numpy as np
import util.AA_Exchange as aa
from pymol import cmd, stored, selector
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import os
import RelaxFastdesign as ros

import argparse
import seaborn as sns
import time
import pyrosetta
pyrosetta.init("-beta -mute all")
import Test_GraphGen as tgg
#-graph gen test statistics for 


#----------------predict_relax------------
modelDirec = '../data/4H_dataset/models/'
straightDirec = '../data/4H_dataset/str_models/'

gg_net = 'data/Best_GraphGen.pt'  
outDirec = '../Figures/GraphGen_Figure/output/gg_new_test'
scoreDirec = '../Figures/GraphGen_Figure/output/'
outName = 'gg_new'  
    
    
def relax_stats(seqList,outName,outDirec,scoreDirec, dOrig= modelDirec, dStraight = straightDirec, scoreList=[],nameList=[]):
    
    labels = ['name','pred_seq','pred_scores','pred_time']

    
    
    for x in range(len(seqList)):
        seq = seqList[x][0]
        name = seqList[x][1]

        #change residues to graphGen prediction
        start = time.time()
        pose = pyrosetta.pose_from_file(f'{dStraight}/{name}_str.pdb')
        pred_pose = ros.mutate_residue(pose,seq)
        pose_relax = ros.fastRelax(pred_pose)
        end = time.time()
        
        pose_relax.dump_pdb(f'{outDirec}/{name}_{outName}.pdb')

        scoreList.append([name, seq, ros.get_pose_scores(pred_pose), end-start])
        nameList.append(name)
        
    tgg.save_scoreList(f'{scoreDirec}/{outName}',nameList=nameList,scoreList=scoreList)
    
    return nameList, scoreList



sG = ps.getSeq(gg_net, direcName = modelDirec,dtest=straightDirec, 
               limit = 100000, hidden=64, mpnn=True,test=True, train=False,catSet=False)
nL, sL = relax_stats(sG,outName,outDirec,scoreDirec)