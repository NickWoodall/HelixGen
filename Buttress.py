#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import scipy
from scipy.stats import norm
import random
import time
import timeit
import math
import localization as lx
import gzip

import util.npose_util as nu
import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree

import tensorflow as tf
import joblib
from sklearn.manifold import MDS
import argparse
from functools import partial
from itertools import starmap,repeat

from pymol import cmd, stored, selector

import GenerateEndpoints as ge
import HelixFit as hf

#reference helix for propogation
zero_ih = nu.npose_from_file('util/zero_ih.pdb')
tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)
stub = tt[7:10].reshape(15,4)


#load distance maps
def load_distance_map(name, dm_file='data/Fits_4H_dm_phi.npz'):
    rr = np.load(dm_file, allow_pickle=True)
    X_train, y_train , featNames = [rr[f] for f in rr.files]
    
    return X_train[y_train==name][:,:-4]


def checkLoss2(testArray,refArray,mask):

    return np.sum(np.square(refArray-testArray)*mask,axis=1)

@tf.function
def maskLoss(y_actual, y_pred,mask):
    custom_loss_val = tf.multiply(mask,tf.square(y_actual-y_pred))
    return custom_loss_val

def buttLoss(recon,mask,refMap,input_z=None,rate=0.05,batch_size=32,cycles=100):

    recon.batch_size = batch_size
    vecDes = recon.mm.transform(np.repeat(refMap.reshape(1,-1) , batch_size, axis=0))
    
    mask =  np.repeat(np.array(mask).reshape(1,-1) , batch_size,axis=0)

    v=tf.convert_to_tensor(vecDes)
    m=tf.convert_to_tensor(mask)
    m2=tf.cast(m, tf.float32)
    v=tf.cast(v,tf.float32)

    if input_z is None:
        input_z = tf.random.uniform(shape=(batch_size,recon.z_size), minval=-1, maxval=1)


    rate = tf.Variable(0.1)
    input_z_var = tf.Variable(input_z)
    g_o = recon.g(input_z_var)
    print('Loss before ')
    print(checkLoss2(g_o,v,m2))

    z=[]
    grads = []

    for t in range(1,cycles):

        #compute Loss
        with tf.GradientTape() as g_tape:
            g_tape.watch(input_z_var)
            g_o = recon.g(input_z_var)
            masked_loss = maskLoss(v,g_o,m2)

        g_grads = g_tape.gradient(masked_loss, input_z_var)
        
        ##### save for when I have to time to check changes
        #this needs to be fixed, ?lower rate? and and move object creation out of loop
        optimizer = tf.keras.optimizers.SGD(learning_rate=rate)
        #optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
        #optimizer = tf.keras.optimizers.Adam()

        optimizer.apply_gradients(zip([g_grads],[input_z_var]))

        if t%10 == 0:
            z.append(input_z_var)
            grads.append(g_grads)

    z.append(input_z_var)
    grads.append(g_grads)
    recon.input_z = input_z_var
    print('Loss after optimization')
    loss_final = checkLoss2(g_o,v,m2)
    print(loss_final)
    #print(f'Reconstruction Error: {sum(recon.reconstructionError()):.2f}')


    return recon, loss_final, z


def buttress1_2_reference_protein(batch,refName='00d94cdcf922f50c6b9c82a8b57d6515_0001',
                                  gen="data/BestGenerator",cycles=1000):
    
    refMap = load_distance_map(refName)
    brec = ge.BatchRecon(gen)
    brec.generate(z=12,batch_size=batch)
    
    mask1   = np.array([1,1,1,0,0,0,0,
                       1,1,0,0,0,0,
                       1,0,0,0,0,
                       0,0,0,0,
                       0,0,0,
                       0,0,
                       0])
    
    
    start = time.time()
    brec, loss_final, z = buttLoss(brec, mask1, refMap,input_z=brec.input_z,batch_size=batch,cycles=cycles)
    end = time.time()
    print('Elapsed time:',end - start)
    brec.generate(z=12,input_z=z[-1],batch_size=batch)
    brec.MDS_reconstruct_()
    brec.reconstructionError()
    brec.to_npose()
    
    return brec, loss_final

def align_helices(ref,pDirec,helix_list=[1,2],outDirec='output/'):
    """Aligns proteins in pDirec (directory) to ref (file) based on helices in helix list [starts at 1]"""
    cmd.delete("all")
    cmd.load(f'{ref}.pdb')
    fileList = os.listdir(pDirec)
    refName = os.path.basename(ref)
    
    for x in fileList:
        cmd.load(f'{pDirec}/{x}')
        cmd.save(f'{outDirec}/buttressTest.pse')
        pairFit_Helix(refName,x[:-4],helix_fits=helix_list) #remove .pdb for pymol
    
    cmd.save(f'{outDirec}/buttressTest.pse')

#visualize 
def get_HelixList(name):
    stored.resi = []
    cmd.iterate_state(1, selector.process(f"{name} and ss 'H' and n. CA"), "stored.resi.append(resi)")

    helixRes = []
    xNow = -1
    for x in stored.resi:
        if int(x)> xNow:
            xNow = int(x)
            helixRes.append([])
        helixRes[-1].append(int(x))
        xNow = xNow+1

    warn = False

    for x in helixRes:
        if len(x) < 4:
            warn=True
    if not len(helixRes) == 4:
        warn=True

    if warn:
        print(f'Check{name}: Helices not as expected')

    return helixRes

def list_alignHelices(name1, name2, helixNum=1):

    hList1 = get_HelixList(name1)
    hList2 = get_HelixList(name2)
    #residues for helix1
    p1_h = hList1[helixNum-1]
    p2_h = hList2[helixNum-1]

    front = True

    while not len(p1_h) == len(p2_h):
        if len(p1_h)>len(p2_h):
            if front:
                p1_h = p1_h[1:]
            else:
                p1_h = p1_h[:-1]

            #front = get_ipython().getoutput('front')
        else:
            if front:
                p2_h = p2_h[1:]
            else:
                p2_h = p2_h[:-1]

            #front = get_ipython().getoutput('front')

    return p1_h, p2_h

def hSel(hListList, name):

    resString = ""

    for x in hListList:
        resString = f'{resString}+{x[0]}-{x[-1]}'

    resString = resString[1:]

    return f'{name} and resi {resString} and name CA'



def pairFit_Helix(prot1,prot2,helix_fits=[1,2]):

    #moves prot2 onto prot1
    pairList1 = []
    pairList2 = []
    
    for x in helix_fits:
        p1_h, p2_h= list_alignHelices(prot1,prot2, helixNum=x)
        pairList1.append(p1_h)
        pairList2.append(p2_h)

    rms = cmd.pair_fit(hSel(pairList2, prot2), hSel(pairList1, prot1))
    return rms


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example Code for search network for Helix3/4 to match input Helix1/2.")

    parser.add_argument("-r","--ref_name", help="Name of Protein from Reference set to get Helix1/2 Distance Map",
                       default='00d94cdcf922f50c6b9c82a8b57d6515_0001')
    parser.add_argument("-c","--cycles", help="Number of gradient descent cycles",
                       default=1000, type=int)
    parser.add_argument("-b","--batch", help="Batch Size",
                       default=8, type=int)
    parser.add_argument("-o","--out_direc", help="directory to output straight helix pdbs",default="output/")
    parser.add_argument("-e","--exclude_align", help="skip align to modeled helices step", action="store_true")
    
    args = parser.parse_args()

    brec, loss_final = buttress1_2_reference_protein(args.batch,refName=args.ref_name,
                                  gen="data/BestGenerator",cycles=args.cycles)



    outDirec = args.out_direc
    for i,c in enumerate(brec.npose_list):
        if loss_final[i]<.005:
            nu.dump_npdb(c,f'{outDirec}build{i}.pdb')

    if not args.exclude_align:
        ref = f'data/4H_dataset/models/{args.ref_name}'
        pDirec = args.out_direc
        align_helices(ref,pDirec,helix_list=[1,2])





