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
gg_net = 'data/Best_GraphGen.pt'

inDirec  = '../Figures/DesignTest/onePer_Looped_b512/'
outDirec  = '../Figures/DesignTest/onePer_Design_b512/'
scoreDirec = '../Figures/DesignTest/'
outName = 'gg_new_onePer'  
    
    
def relax_stats(seqList,outName,outDirec,scoreDirec, inDirec=inDirec, scoreList=[],nameList=[]):
    
    labels = ['name','pred_seq','pred_scores','pred_time']

    
    
    for x in range(len(seqList)):
        seq = seqList[x][0]
        name = seqList[x][1]

        #change residues to graphGen prediction
        start = time.time()
        pose = pyrosetta.pose_from_file(f'{inDirec}/{name}.pdb')
        pred_pose = ros.mutate_residue(pose,seq)
        pose_relax = ros.fastRelax(pred_pose)
        end = time.time()
        
        pose_relax.dump_pdb(f'{outDirec}/{name}_{outName}.pdb')

        scoreList.append([name, seq, ros.get_pose_scores(pred_pose), end-start])
        nameList.append(name)
        
    tgg.save_scoreList(f'{scoreDirec}/{outName}',nameList=nameList,scoreList=scoreList)
    
    return nameList, scoreList



sP = ps.getSeq(gg_net, direcName = inDirec,limit = 10000, hidden=64, mpnn=True)
nL, sL = relax_stats(sP,outName,outDirec,scoreDirec)