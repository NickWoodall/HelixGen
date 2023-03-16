from __future__ import print_function
import os, time, gzip
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from collections import defaultdict
import urllib.request, json 
#import requests as reqs
import time
import timeit
import copy
import json, time, os, sys, glob
import shutil

from pymol import stored, selector, cmd
from weakref import WeakKeyDictionary

import pickle

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

sys.path.insert(0,f'experiments/')
sys.path.insert(0,f'struct2seq/')
import struct2seq
import utils # from graphgen experiments folder
import noam_opt
import PDBDataset
import util.gg_util as gg #my selected graph gen functions
import util.AA_Exchange as aa
import argparse



def setup_device_rng_new(seed=1111):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    return device

def getSeq(checkpoint_path, direcName = '../data/BCov_4H_dataset/BCov_Models_Straight/',dtest=None, limit = 100000, hidden=64, mpnn=True,
          test=False,train=False,catSet=False):
    """Uses graph gen model to predict sequence"""
    #load model
    #dtest is for getting data for testing graph gen network with training dataset
    dev = setup_device_rng_new(seed=1111)
    mod = model = struct2seq.Struct2Seq(
            num_letters=20, 
            node_features=hidden,
            edge_features=hidden, 
            hidden_dim=hidden,
            k_neighbors=30,
            protein_features="full",
            dropout=0.1,
            use_mpnn=mpnn
        ).to(dev)
    state_dicts = torch.load(checkpoint_path, map_location='cpu')
    mod.load_state_dict(state_dicts['model_state_dict'])
    optimizer = noam_opt.get_std_opt(mod.parameters(), hidden)
    criterion = torch.nn.NLLLoss(reduction='none')

    if test:
        sd = PDBDataset.StructureDataset_PDBDirec(direcName,dStraight=dtest,limit=limit, clusterLoad='../Clustering/data/refData.npz',test=True)
    elif train:
        sd = PDBDataset.StructureDataset_PDBDirec(direcName,dStraight=dtest,limit=limit, clusterLoad='../Clustering/data/refData.npz',test=False)
    else:
        sd = PDBDataset.StructureDataset_PDBDirec(direcName,limit=limit)
        
    sd.data = sorted(sd.data, key = lambda i: i['name']) #sort data by filename
    
    seqList = []

    REPEATS = 1
    temp = 0.1
    total_step = 0
    total_residues = 0
    start_time = time.time()
    # Validation epoch
    mod.eval()
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        for ix, protein in enumerate(sd):

            prot_clones = [copy.deepcopy(protein) for i in range(REPEATS)]
            X, S, mask, lengths = utils.featurize(prot_clones, dev)

            #X, S, mask, lengths = featurize(protein, dev)
            native_seq = gg._S_to_seq(S[0], mask[0])
            S_sample = mod.sample(X, lengths, mask, temperature=temp)
            #print('')
            #print('name',protein['name'])
            #print(native_seq)

            # Compute scores
            log_probs = mod(X, S_sample, lengths, mask)
            scores = gg._scores(S_sample, log_probs, mask,criterion)
            scores = scores.cpu().data.numpy()

            seq = gg._S_to_seq(S_sample[0], mask[0])
            for b_ix in range(REPEATS):
                seq = gg._S_to_seq(S_sample[b_ix], mask[0])
                #print(seq)

            seqList.append([seq,protein['name']])
            
    
    return seqList


def getSeq_fileList(checkpoint_path, fileList, direcName = '../data/BCov_4H_dataset/BCov_Models_Straight/',hidden=64,mpnn=True):
    """Uses graph gen model to predict sequence"""
    #load model
    dev = setup_device_rng_new(seed=1111)
    mod = model = struct2seq.Struct2Seq(
            num_letters=20, 
            node_features=hidden,
            edge_features=hidden, 
            hidden_dim=hidden,
            k_neighbors=30,
            protein_features="full",
            dropout=0.1,
            use_mpnn=mpnn
        ).to(dev)
    state_dicts = torch.load(checkpoint_path, map_location='cpu')
    mod.load_state_dict(state_dicts['model_state_dict'])
    optimizer = noam_opt.get_std_opt(mod.parameters(), hidden)
    criterion = torch.nn.NLLLoss(reduction='none')

    sd = PDBDataset.StructureDataset_PDBFile(fileList,direcName=direcName)
    
    seqList = []

    REPEATS = 1
    temp = 0.1
    total_step = 0
    total_residues = 0
    start_time = time.time()
    # Validation epoch
    mod.eval()
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        for ix, protein in enumerate(sd):

            prot_clones = [copy.deepcopy(protein) for i in range(REPEATS)]
            X, S, mask, lengths = utils.featurize(prot_clones, dev)

            #X, S, mask, lengths = featurize(protein, dev)
            native_seq = gg._S_to_seq(S[0], mask[0])
            S_sample = mod.sample(X, lengths, mask, temperature=temp)
            #print('')
            #print('name',protein['name'])
            #print(native_seq)

            # Compute scores
            log_probs = mod(X, S_sample, lengths, mask)
            scores = gg._scores(S_sample, log_probs, mask,criterion)
            scores = scores.cpu().data.numpy()

            seq = gg._S_to_seq(S_sample[0], mask[0])
            for b_ix in range(REPEATS):
                seq = gg._S_to_seq(S_sample[b_ix], mask[0])
                #print(seq)

            seqList.append([seq,protein['name']])
            
    
    return seqList



def pdbSeq(fname,direc=''):


    name = f'{direc}{fname}'

    seqActual = []

    cmd.load(f'{name}.pdb')
    #get sequence
    stored.resn = []
    cmd.iterate_state(1, selector.process(f"{name} and n. CA"), "stored.resn.append(resn)")
    seq = ''
    for x in stored.resn:
        seq = f'{seq}{aa.aaCodeExchange(x)}'

    return seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--pdbdirec", help="directory to load pdb files from", default="../data/4H_dataset/models/")
    parser.add_argument("-n", "--numStruct", help="limits number of structures to load. Default 1000", default=1000, type=int)
    parser.add_argument("-g", "--generator", help="generator model to load", default="data/Best_GraphGen.pt")
    parser.add_argument("-o", "--outputFile", help="output as text sequence, name newline",default="output/seqOut.txt")
    
    if not os.path.exists("output/"):
        os.makedirs("output/")
    
    args = parser.parse_args()
    
    seq1 = getSeq(args.generator,direcName = args.pdbdirec, limit=args.numStruct)
    
    with open(f'{args.outputFile}',"w") as f:
        for x in seq1:
            f.write(f'{x[0]},{x[1]}\n')






