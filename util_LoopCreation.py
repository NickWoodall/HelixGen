from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
import sys
import util.npose_util as nu
import datetime
import HelixFit as hf

import util.RotationMethods as rm
import util.npose_util as nu
import numpy as np
import FitTransform as ft

import argparse

#reference pdb file
zero_ih = nu.npose_from_file('util/zero_ih.pdb')
#reformatted into each amino acid containin its component atoms
tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)
#3 atom stubs for building
stub = tt[7:10].reshape(15,4)

#list of loop atom positions, #all loops have 4 helical residues on each side
# rr = np.load('data/all_loops.npz', allow_pickle=True)
# all_loops = [rr[f] for f in rr.files][0]

#list of loops fitted with helical extensions,
#one vector from end of helix 1, to beginning of helix2 (loop vector) [3 features] 0-2
#one vector from beginning of helix2 to the end of helix 2 [3 features] 3-5
#the helix vector is normalized to one
#also includes phi value change form helix1 to 2 [1 feature1] 6
#loop name feature 7 [int number]

# rr = np.load('data/loopFeat_helixnorm.npz',allow_pickle=True)
# loopFeat_Actual = [rr[f] for f in rr.files][0] 

#list of loops fitted with helical extensions,
#one vector from end of helix 1, to beginning of helix2 (loop vector)[3 features] 0-2
#one vector from beginning of helix2 to the end of helix 2 [3 features] 3-5
#both vectors normalized
#also includes phi value change form helix1 to 2 [1 feature1] 6 
#loop name feature 7 [int number]

# rr = np.load('data/loopFeat_bothnorm.npz',allow_pickle=True)
# loopFeat_Tree = [rr[f] for f in rr.files][0] #


def align_loop(build,loop):
    """returns loop aligned to end of build, overlapping"""
    tpose1 = nu.tpose_from_npose(loop)
    tpose2 = nu.tpose_from_npose(build)

    itpose1 = np.linalg.inv(tpose1)
    
    # align first residue of loop, loop[0], to last residue of current build[-1]
    xform = tpose2[-1] @ itpose1[0]

    aligned_npose1 = nu.xform_npose( xform, loop )
    
    #remove overlap and return
    return aligned_npose1[5:]

def extend_helix(build, res):
    """Extend straight helix up to 20 residues"""
    #limited by reference helix length

    ext = align_loop(build,zero_ih)
    
    if res > 20:
        return ext
    
    return ext[:(res*5)]


# In[5]:


#methods to load txt file of loops and resave as numpy file


def saveAsNumpy(np_in,name,direc='data/'):
    """Simple reminder function to how to save as numpy."""
    np.savez_compressed(f'{direc}{name}.npz',np_in)

def numpyString(arrayIn):
    """Fix Numpy String ouput to save as csv."""
    x = np.array2string(arrayIn,separator=",")
    x = x.replace('[','')
    x = x.replace(']','')
    return x

def save_as_csv_chunks(arrayIn,name,direc='csv_datasets/'):
    """Save Chunk csv, into loop based seperation of atom vectors"""
    
    with open(f'{direc}{name}','w') as f:
        for x in arrayIn:
            f.write(',\n')#initial comma separates each loop
            f.write(numpyString(x))
            f.write('\n')
            
def load_csv_chunks(fname,direc='data/csv_datasets/'):
    """Load atom positions vectors by their approriate loops separations"""
    npList = []
    with open(f'{direc}{fname}.txt','r') as f:
        a = f.readlines()

    chunkList = []

    for i,c in enumerate(a):
        if c[0] == ',':
            chunkList.append(i)
    chunkList.append(len(a))
            

    for x in range(0,len(chunkList)-1):
        strChunk = "" 
        for y in a[chunkList[x]+1:chunkList[x+1]]:
            strChunk = f'{strChunk}{y}'
        npList.append(np.fromstring(str(strChunk), dtype=float, sep=",").reshape((-1,4)))
        
    return np.array(npList,dtype='O')

def transfer_loops_from_csv_to_numpy(name='all_loops', loopFile='all_loops_bCov.txt',direcCSV='data/csv_datasets/',
                                    direcOut='data/'):
    
    loopFile = loopFile[:-4]
    numpy_array_loops = load_csv_chunks(loopFile,direc=direcCSV)
    
    saveAsNumpy(numpy_array_loops,name,direc=direcOut)
    


# In[6]:


def generate_loops_with_helices(loopFile = 'data/all_loops.npz',hL=10,outDirec='data/bCov_4H_dataset/BCov_LoopsToFit/'):
    """Extend straight helices on either side of loop for fitting with HelixFit."""
    #hL = helix length
    rr = np.load(loopFile, allow_pickle=True)
    all_loops = [rr[f] for f in rr.files][0]
    hL = 10
    
    #generate all loops
    for i,c in enumerate(all_loops):
        build = stub.copy()

        h1=extend_helix(build,hL)
        b1 = np.append(build,h1,0)
        l1 = align_loop(b1,c)

        b2 = np.append(b1,l1,0)
        h2 = extend_helix(b2,hL+3) #match stub
        b3 = np.append(b2,h2,0)
        nu.dump_npdb(b3,f'{outDirec}/loop_{i}.pdb')
    
    
def fit_loops(name='loopFits', loopDirec='data/bCov_4H_dataset/BCov_LoopsToFit/', outDirec='data/'):
    """Fit helical extended loops. Save as appended .csv file"""
    
    fileList = os.listdir(loopDirec)
    h1 = hf.HelicalProtein(fileList[0],direc=loopDirec,name=fileList[0][:-4],expected_helices=2)
    h1.fit_all()
    
    with open(f'{outDirec}/{name}.csv','w') as f:
        f.write(h1.getLabel_())
        f.write('\n')
        
    for i,c in enumerate(fileList):
    
        h1 = hf.HelicalProtein(c,direc=loopDirec,name=c[:-4],expected_helices=2)
        h1.fit_all()

        fitString = h1.export_fits_()
        with open(f'{outDirec}/{name}.csv','a') as f:
            f.write(f'{fitString}\n')

        if i%1000 ==0:
            print(f'{i} fits done')




def convert_fit_to_endpoints(loopFit_file='data/loopFits.csv', expected_helices=2):
    """Convert from HelicalFit to features for Loop Endpoints Methods: for KDTree. See comments"""
    #list of loops fitted with helical extensions,
    #one vector from end of helix 1, to beginning of helix2 (loop vector) [3 features] 0-2
    #one vector from beginning of helix2 to the end of helix 2 [3 features] 3-5
    #the helix vector is normalized to one
    #also includes phi value change form helix1 to 2 [1 feature1] 6
    #loop name feature 7 [int number]
    
    dfRead = pd.read_csv(loopFit_file)
    df1 = ft.prepData_Str(dfRead,rmsd_filter=100)
    df2 = ft.EndPoint(df1,num_helices=expected_helices)
    df1['delta_phi'] = df1['phi1_2']-df1['phi1_1']
    phi_loops = df1.to_numpy()
    loop_fit_ep = df2.to_numpy()
    
    #get names [correspond to loop values in ]
    nameList = []
    for x in df1['name']:
        nameList.append(int(x.split("_")[1]))
        
    names = np.array(nameList)
    loopVec = loop_fit_ep[:,6:9]-loop_fit_ep[:,3:6]

    nextHelixVec = np.zeros(loopVec.shape)
    for x in range(len(loop_fit_ep)):
        nextHelixVec[x,:] = rm.normalize(loop_fit_ep[x,9:12] - loop_fit_ep[x,6:9])
    delta_phi = loop_fit_ep[:,-2]-loop_fit_ep[:,-3] #phi values for each helix
    
    normLoopVec = np.zeros(loopVec.shape)
    for x in range(len(loop_fit_ep)):
        normLoopVec[x,:] = rm.normalize(loop_fit_ep[x,6:9] - loop_fit_ep[x,3:6])
        
    loop_Features = np.hstack((loopVec,nextHelixVec,delta_phi.reshape(-1,1))).astype(dtype=np.float32)
    loop_Feats = np.hstack((loop_Features,names.reshape(-1,1)))
    
    loop_Features_twoNorm = np.hstack((normLoopVec,nextHelixVec,delta_phi.reshape(-1,1))).astype(dtype=np.float32)
    loop_Feats_twoNorm = np.hstack((loop_Features_twoNorm,names.reshape(-1,1)))
    
    
    inds = np.argsort(names)
    loopFeats = loop_Feats[inds]
    loopFeats_twoNorm = loop_Feats_twoNorm[inds]
    
    return loopFeats, loopFeats_twoNorm
    
    
def save_loopFeats(lF,lF_twoNorm, outDirec='data/', nameOut='LoopFeats'):
    """Save Loop Features with helix vector normalized (features 3-5) and all features normalized"""
    np.savez_compressed(f'{outDirec}{nameOut}_helixnorm.npz',lF[:,:7])
    np.savez_compressed(f'{outDirec}{nameOut}_bothnorm.npz',lF_twoNorm[:,:7])
    

    
    
    

if __name__ == "__main__":
    #
    
    parser = argparse.ArgumentParser(description="Generates Loop numpy saves (full atom/features) for other scripts. Not specifying flag results in default names put in data/ directory.")
    
    
    parser.add_argument("-d","--direc", help= "Directory to load input data." )
    parser.add_argument("-o","--outdirec", help= "Directory to output data." )
    parser.add_argument("-i","--infile", help= "Input file to load")
    parser.add_argument("-n","--name", help= "output file name")
    
    
    
    
    parser.add_argument("-a","--loop_csv_to_numpy", help="CSV of loop atoms to numpy array save, requires -n,-d,-o-i", action="store_true")
    parser.add_argument("-f","--fit_loops", help="fits helices on either side of a loop with Helical Fit. requires-n,-d,-o", action="store_true")
    parser.add_argument("-g","--generateLoopsHelices", help="extend helices on either side of loops for fitting. requires -i and -o", action="store_true")
    parser.add_argument("-m","--make_features", help="turn loop fits into features for the kdtree. -i, o, n", action="store_true")
    parser.add_argument("-r","--remake", help= "remake all by defaults names saving to data folder. No other flags. Will take a few hours.", action = "store_true")
    parser.add_argument("-j","--just_functional", help= "do not replicate the loop fits, but regenerate necessary date from csv", action = "store_true")
    args = parser.parse_args()
    
    
    if args.loop_csv_to_numpy:
        #requires -n,-d,-o-i
        if args.direc and args.outdirec and args.infile and args.name:
            transfer_loops_from_csv_to_numpy(name=args.name, loopFile=args.infile,direcCSV=args.direc,
                                            direcOut=args.outdirec)
    elif args.generateLoopsHelices:
        #requires -i and -o
        if args.infile and args.outdirec:
            generate_loops_with_helices(loopFile = args.infile,hL=10,outDirec=args.outdirec)
            
    elif args.fit_loops:
        #requires
        if args.outdirec and args.name and args.direc:
            fit_loops(name=args.name, loopDirec=args.direc,outDirec=args.outdirec)
            
    elif args.make_features:
        #requires -i, o, n
        if args.outdirec and args.name and args.infile:
            loopFeats, loopFeats_twoNorm = convert_fit_to_endpoints(loopFit_file=args.infile, expected_helices=2)
            save_loopFeats(loopFeats,loopFeats_twoNorm, outDirec=args.outdirec, nameOut=args.name)
        
        
    
    
    
    elif args.remake:
    
        transfer_loops_from_csv_to_numpy(name='all_loops', loopFile='loop_struct.txt',direcCSV='data/',
                                            direcOut='data/')
        generate_loops_with_helices(loopFile = 'data/all_loops.npz',hL=10,outDirec='data/4H_dataset/LoopsToFit/')

        fit_loops(name='LoopFits', loopDirec='data/4H_dataset/LoopsToFit/',outDirec='data/')

        loopFeats, loopFeats_twoNorm = convert_fit_to_endpoints(loopFit_file='data/loopFits.csv', expected_helices=2)

        save_loopFeats(loopFeats,loopFeats_twoNorm, outDirec='data/', nameOut='LoopFeats')
        
    elif args.just_functional:
    
        transfer_loops_from_csv_to_numpy(name='all_loops', loopFile='loop_struct.txt',direcCSV='data/',
                                            direcOut='data/')
                                            
        loopFeats, loopFeats_twoNorm = convert_fit_to_endpoints(loopFit_file='data/loopFits.csv', expected_helices=2)

        save_loopFeats(loopFeats,loopFeats_twoNorm, outDirec='data/', nameOut='LoopFeat')
    






