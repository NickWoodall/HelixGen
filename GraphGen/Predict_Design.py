import sys


import PredictSeq as ps
import pyrosetta
import RelaxFastdesign as ros
import time
import numpy as np
import util.AA_Exchange as aa
from pymol import cmd, stored, selector
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import os

import argparse
import seaborn as sns



#------------------ AA sequence prediction only code ------------------------




def get_pdb_seq(seqList,direcName='../data/bCov_4H_dataset/BCov_Models/',removeEnding=True):
    """record actual sequence from designed protein"""
    seqActual = []

    for x in range(len(seqList)):
        if removeEnding:
            name = seqList[x][1][:-4] #removes .pdb and _str
        else:
            name = seqList[x][1]
        cmd.delete("all")
        cmd.load(f'{direcName}/{name}.pdb')

        #get sequence
        stored.resn = []
        cmd.iterate_state(1, selector.process(f"{name} and n. CA"), "stored.resn.append(resn)")
        seq = ''
        for x in stored.resn:
            seq = f'{seq}{aa.aaCodeExchange(x)}' #convert AA format, single, triple letter, full name etc

        seqActual.append(seq)

    return seqActual

#------------ Code to call Rosetta methods for desiging proteins using the GraphGen Network -----------


def mutate_relax(sP, direc = '', outName='out', outDirec='output/'):
    
    
    for x in range(len(sP)):
        pred_seq = sP[x][0]
        name = sP[x][1]

        #change residues to graphGen prediction
        pose = pyrosetta.pose_from_file(f'{direc}/{name}.pdb')
        pred_pose = ros.mutate_residue(pose,pred_seq)

        #relax using predicted sequence this work
        pred_pose_relax = ros.fastRelax(pred_pose)
        
        pred_pose_relax.dump_pdb(f'{outDirec}/{name}_{outName}.pdb')
    
               


straightDirec = '../data/4H_dataset/str_models/'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Use the GraphGen network to design AAs for straight 4H bundles.")
    
    parser.add_argument("-i", "--inDirec", help="Input Structure Directory",default=straightDirec)
    parser.add_argument("-o", "--outDirec", help="Directory to output pdb files.", default = 'output/')
    parser.add_argument("-g", "--graphGen", help="Location of GraphGen Network", default='data/Best_GraphGen.pt')
    parser.add_argument("-n", "--outName", help="Name to append to differentiate output")
    
    args = parser.parse_args()

    pyrosetta.init("-beta -mute all")
    
    if not os.path.exists(args.outDirec):
        os.makedirs(args.outDirec)
    
    
    sP = ps.getSeq(args.graphGen, direcName = args.inDirec, hidden=64)
    mutate_relax(sP,direc=args.inDirec,outName=args.outName, outDirec=args.outDirec)