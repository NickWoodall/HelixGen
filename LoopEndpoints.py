import os
import sys

from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign
import numpy as np

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

import util.npose_util as nu
import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import joblib
from sklearn.manifold import MDS
import argparse
from functools import partial
from itertools import starmap,repeat

import GenerateEndpoints as ge

def rotate_fit(feat,angle):
    """Rotates the Endpoints vectors around the z-axis to make initial addition a random phi value"""
    
    feat_vec = feat.copy()
    
    xform1 = nu.xform_from_axis_angle_rad(np.array([0,0,-1]),angle)
    
    oneCol = np.ones((feat_vec.shape[0],1))

    vecTranslation = np.hstack((feat_vec[:,:3],oneCol))
    vecNextEnd = np.hstack((feat_vec[:,3:6],oneCol))
    
    vecOut1  = nu.xform_npose(xform1, vecTranslation)
    
    vecOut2 = []
    
    for ind,i in enumerate(vecTranslation):

        totVec = i+vecNextEnd[ind]
        totVec[3] = 1
        
        inter  = nu.xform_npose(xform1,totVec)
        
        outer = inter[0]-vecOut1[ind]
        
        vecOut2.append(outer)
        
    vecOut2 = np.array(vecOut2)
    
    phi = feat[:,-1] + angle
    phi = phi.reshape(-1,1)
    
    return np.hstack((vecOut1[:,:-1],vecOut2[:,:-1],phi))



def create_kdTree(features):

    rotFeat = features.copy()

    for x in range(36):
        rotFeat = np.vstack((rotFeat,rotate_fit(features,x*np.pi/36)))

    #transform phi values into the same range as the normalized vector pieces
    mm1 = MinMaxScaler((-1, 1))
    mm2 = MinMaxScaler((-1, 1))

    rotFeat[:,-1] = mm1.fit_transform(rotFeat[:,-1].reshape(-1, 1)).reshape(-1)
    features[:,-1] = mm2.fit_transform(features[:,-1].reshape(-1, 1)).reshape(-1)
    
    binTreePhi = cKDTree(rotFeat,balanced_tree=False)
    binTreePhiS = cKDTree(features,balanced_tree=False) #small tree ( not rotated)
    
    return binTreePhi, binTreePhiS


# In[3]:


zero_ih = nu.npose_from_file('util/zero_ih.pdb')
tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)
stub = tt[7:10].reshape(15,4)


#list of loop atom positions, #all loops have 4 helical residues on each side

rr = np.load('data/all_loops.npz', allow_pickle=True)
# #rr = np.load('data/all_loops_bCov.npz', allow_pickle=True)
all_loops = [rr[f] for f in rr.files][0]

rr = np.load('data/loopFeat_helixnorm.npz',allow_pickle=True)
loopFeat_Actual = [rr[f] for f in rr.files][0] 

rr = np.load('data/loopFeat_bothnorm.npz',allow_pickle=True)
loopFeat_Tree = [rr[f] for f in rr.files][0] #

binTreePhi, binTreePhiS = create_kdTree(loopFeat_Tree)

feats = loopFeat_Tree
feats_acc = loopFeat_Actual


#------------ align/build and analyze fragments---------------------
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

ref = extend_helix(stub.copy(),10) #reference for fitted loops
        
    
def whole_prot_clash_check(npose,hList,threshold=2.85):
    """Checks for clashes between helices, given list of where in residues helices start and stop."""
    
    indexList = []
    curI = 0
    
    #get helical indices in npose form
    for ind,i in enumerate(hList):
        indexList.append(list(range(curI,curI+i*5)))
        curI += i*5
        
    fullSet = set(range(len(npose)))
    
    clashedCount = 0
    
    #remove 1 helix at a time and check it for clashing with the other three
    for ind,i in enumerate(indexList):
        build = list(fullSet.difference(set(i)))
        clashedCount += check_clash(npose[build],npose[i],threshold)
        
    return clashedCount
        
def check_clash(build_set, query, threshold=2.85):
    """Return True if new addition clashes with current set"""
    
    #if null, re
    if len(build_set) <= 5 or len(query) <= 5:
        return True
    query_set = query[5:]
    seq_buff = 5 # +1 from old clash check, should be fine
    if len(query_set) < seq_buff:
        seq_buff = len(query_set)
    elif len(build_set) < seq_buff:
        seq_buff = len(build_set)

    axa = scipy.spatial.distance.cdist(build_set,query_set)
    for i in range(seq_buff):
        for j in range(seq_buff-i):
            axa[-(i+1)][j] = threshold + 10 # moded from .1 here
            

    if np.min(axa) < threshold: # clash condition
        return True

    return False

def get_neighbor_2D(build):
    """Return 2D Neighbor Matrix, for slicing later"""
    
    pose = build.reshape(int(len(build)/5),5,4)

    ca_cb = pose[:,1:3,:3]
    conevect = (ca_cb[:,1] - ca_cb[:,0] )
    # conevect_lens = np.sqrt( np.sum( np.square( conevect ), axis=-1 ) )
    # for i in range(len(conevect)):
    #     conevect[i] /= conevect_lens[i]

    conevect /= 1.5

    maxx = 11.3
    max2 = maxx*maxx

    neighs = np.zeros((len(ca_cb),(len(ca_cb))))

    core = 0
    surf = 0

    summ = 0
    for i in range(len(ca_cb)):

        vect = ca_cb[:,0] - ca_cb[i,1]
        
        vect_length2 = np.sum( np.square( vect ), axis=-1 )

        ind = np.where((vect_length2 < max2) | (vect_length2 > 4))[0]
        vect_length = np.sqrt(vect_length2)

        vect = np.divide(vect,vect_length.reshape(-1,1))

        # bcov hack to make it ultra fast
        # linear fit to the above sigmoid
        dist_term = np.zeros(len(vect))

        for j in ind:
            if ( vect_length[j] < 7 ):
                dist_term[j] = 1
            elif (vect_length[j] > maxx ):
                dist_term[j] = 0
            else:
                dist_term[j] = -0.23 * vect_length[j] + 2.6

        angle_term = ( np.dot(vect, conevect[i] ) + 0.5 ) / 1.5

        for j in ind:
            if ( angle_term[j] < 0 ):
                angle_term[j] = 0
        neighs[i] = dist_term * np.square( angle_term )

    return neighs

def get_scn(sc_matrix, indices=None, percent_core = True):
    """Returns percent of residues that are in the core of the protein"""
    #core is defined as having greater than 5.2 summed from neighbor matrix
    
    if indices:
        indices = np.array(indices,dtype=np.int32)
        summed = np.sum(sc_matrix[indices], axis= -1)
    else:
        indices = np.array(list(range(len(sc_matrix))))
        summed = np.sum(sc_matrix,axis=-1)
        
    if percent_core:
        out = (summed > 5.2).sum() / len(indices)
    else:
        #av_scn
        out = np.mean(summed)
    
    return out


# In[5]:


#---------------creation of kdtree from loop library------------------

#Brian Coventry's Helical Loop Library - 4 helical residues - short loop - 4 helical residues
#Each loop was extended by 10 AA via straight helix, First Helix aligned on the Z-axis
#Fit to straight helices fits with phi values for phase, helix phase independent from z-value
#Converted to endpoints representations (4 endpoints) and then reduced to vector between 
#second and third endpoint (loop) and third and fourth endpoint (second helix) + helical phase of second helix
#normalized to 0 to 1
#min and mix for last column [phase change] is -pi and +pi changed to -1 to 1


import HelixFit as hf
import FitTransform as ft
import util.RotationMethods as rm


def loop_fit_protein_create(hL=10):
    """Create a loop with helices on either side for fitting loop library."""
    for i,c in enumerate(all_loops):
        build = stub.copy()

        h1=extend_helix(build,hL)
        b1 = np.append(build,h1,0)
        l1 = align_loop(b1,c)

        b2 = np.append(b1,l1,0)
        h2 = extend_helix(b2,hL+3) #match stub
        b3 = np.append(b2,h2,0)
        nu.dump_npdb(b3,f'data/bCov_4H_dataset/BCov_LoopsToFit/loop_{i}.pdb')
        
def hfit_loop_proteins(csvFile = 'data/loopFits_new.csv',dataDirec='data/bCov_4H_dataset/BCov_LoopsToFit/'):
    """Fits Loop Proteins and saves data"""
    
    fileList = os.listdir(dataDirec)
    h1 = hf.HelicalProtein(fileList[0],direc=dataDirec,name=fileList[0][:-4],expected_helices=2)
    h1.fit_all()

    with open(csvFile,'w') as f:
        f.write(h1.getLabel_())
        f.write('\n')
    
    for i,c in enumerate(fileList):

        h1 = hf.HelicalProtein(c,direc=dataDirec,name=c[:-4],expected_helices=2)
        h1.fit_all()

        fitString = h1.export_fits_()
        with open('data/loopFits.csv','a') as f:
            f.write(f'{fitString}\n')

        if i%1000 ==0:
            print(f'{i} fits done')
            
def convertFitstoNumpy(csvFile = 'data/loopFits.csv'):
    dfRead = pd.read_csv(csvFile)
    df1 = ft.prepData_Str(dfRead,rmsd_filter=100)
    df2 = ft.EndPoint(df1,num_helices=2)
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
        
    normLoopVec = np.zeros(loopVec.shape)
    for x in range(len(loop_fit_ep)):
        normLoopVec[x,:] = rm.normalize(loop_fit_ep[x,6:9] - loop_fit_ep[x,3:6])
        
    delta_phi = loop_fit_ep[:,-2]-loop_fit_ep[:,-3] #phi values for each helix
    
    loop_Features = np.hstack((loopVec,nextHelixVec,delta_phi.reshape(-1,1))).astype(dtype=np.float32)
    loop_Feats = np.hstack((loop_Features,names.reshape(-1,1)))
    
    loop_Features_twoNorm = np.hstack((normLoopVec,nextHelixVec,delta_phi.reshape(-1,1))).astype(dtype=np.float32)
    loop_Feats_twoNorm = np.hstack((loop_Features_twoNorm,names.reshape(-1,1)))
    
    inds = np.argsort(names)
    
    #sort to keep in save order as all loops vector
    loopFeats_twoNorm = loop_Feats_twoNorm[inds]
    loopFeats = loop_Feats[inds]
    
    def saveAsNumpy(np_in,name,direc='data/'):
        feats = np_in[:,:7] #remove name
        np.savez_compressed(f'{direc}{name}.npz',feats=feats)
        
    saveAsNumpy(loopFeats,'loopFeat_helixnorm')
    saveAsNumpy(loopFeats_twoNorm,'loopFeat_bothnorm')





def angle_two_vectors(v1,v2):
        #assuming normalize
        #https://onlinemschool.com/math/library/vector/angl/

        # cos α = 	a·b
        #         |a|·|b|

        dp = np.dot(v1,v2)
        return  acos(dp)

def return_aligned(ep,hnum=0):
    
    #align first helix to the z-axis
    
    #p1 = hstart, p2=hend, p3=nextHstart, p4= nextHend
    
    end_points = ep.copy()
    
    #p1 = ep[hnum*2]
    #p2 = ep[hnum*2+1]
    #p3 = ep[(hnum+1)*2]
    #p4 = ep[(hnum+1)*2+1]
    
    #move p1 to origin
    #p4 = p4-p1
    #p3 = p3-p1
    #p2 = p2-p1
    #p1 = p1-p1
    
    end_points = end_points - end_points[0]
    
    p1 = end_points[hnum*2]
    p2 = end_points[hnum*2+1]
    
    zUnit  = np.array([0,0,-1])
    vector = normalize(p2 - p1)

    ang = angle_two_vectors(vector,zUnit)
    axisRot = normalize(np.cross(vector,zUnit))
    aRot=np.hstack((axisRot,[1]))
    xform1 = nu.xform_from_axis_angle_rad(aRot,ang)

    
    #pVec = np.vstack((p1,p2,p3,p4))
    oneCol = np.ones((end_points.shape[0],1))
    pVec = np.hstack((end_points,oneCol))

    newPVec = nu.xform_npose(xform1,pVec)
    
    return newPVec

def get_dist(pVec):
    dist = []
    for x in range(0,len(pVec)-1):
        dist.append(np.linalg.norm(pVec[x+1][:3]-pVec[x][:3]))
        
    return dist
        

def get_query(pVec,hnum):
    """Return Kdtree query from endpoints"""
    
    ep1 = hnum*2
    
    #okay here, careful normalizing vector with one on the end
    vec1 = normalize(pVec[ep1+2][:3]-pVec[ep1+1][:3])
    vec2 = normalize(pVec[ep1+3][:3]-pVec[ep1+2][:3])
        
    return np.hstack((vec1,vec2))
    
def get_query_true(ep_guide, ep_true, hnum):
    """Return KdTree query accounting for offset between the actual build(true) and the guide endpoints."""
    
    ep1 = hnum*2
    
    #okay here, careful normalizing vector with one on the end, since sub makes it zero
    vec1 = normalize(ep_guide[ep1+2][:3]-ep_true[ep1+1][:3])
    vec2 = normalize(ep_guide[ep1+3][:3]-ep_guide[ep1+2][:3])
    return np.hstack((vec1,vec2))
    
def convert_index(index):
    """Converts index from expanded (rotated Loop Fit library) to original (before rotation)"""
    
    base = int(index/len(feats))*len(feats)
    return index-base

def rotate_ep(ep,index,extNum):
    """Rotates endpoints to match phi-value of first loop addition"""
    #takes endpoints, loop index and number of residues and rotates into proper frame
    
    #10 was used for the fits, binTree Library therefore Ref
    #3 added for stubs
    
    offset = (extNum-10)*1.74533 #100deg in radians for 3.6 residues per turn. mod -4
    
    
    v1 = feats[index][:3] #remove convert? feats[convert_index(index)]
    v2 = normalize(ep[2][:3]-ep[1][:3])
    
    v1[2]=0
    v2[2]=0
    
    ang = angle_two_vectors(normalize(v1),v2)
    ang = ang-offset
    axisRot = np.array([0,0,-1,1])
    xform1 = nu.xform_from_axis_angle_rad(axisRot,-ang)
    
    newPVec = nu.xform_npose(xform1,ep)
    
    return newPVec
    
    
def helixLength_indices(indexList,hnum=0):
    """Return the indices for the each helix length in the index list. 
    So that the corresponding build list [atom xyz] can be expanded correctly."""
    
    hnum_ind = hnum*2
    
    hL = np.unique(indexList[:,hnum_ind])
    helix_indices = np.array(list(map(lambda x: indexList[:,hnum_ind]==x,hL)))
    
    return helix_indices

def get_transform(target,mobile):
    """Get Rotation Matrix that aligns the mobile piece to the target fragment"""
    # transform returns loop aligned to end of build, overlapping
    tpose1 = nu.tpose_from_npose(mobile)
    tpose2 = nu.tpose_from_npose(target)

    itpose1 = np.linalg.inv(tpose1)
    
    # loop[0] to build[-1]
    xform = tpose2[-1] @ itpose1[-1]

    return xform
    

def forgeAhead(ar1,ar2_len):
    """If the mask array will delete the full list, exit program"""
    
    if np.sum(np.invert(ar1)) == ar2_len:
        return False
    else:
        return True
    

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm



def random_reduce(arrayList, num_to_keep = 20):
    
    if num_to_keep > arrayList.shape[0]:
        num_to_keep =  arrayList.shape[0]
    
    indexer = np.random.choice(range(arrayList.shape[0]),num_to_keep ,replace=False)
    
    return indexer
    
    


# In[6]:


def first_helix(end_points,length_mod=1):
    
    #generic helical AA to extend from
    build = stub.copy()
    
    # -4 to account for single stub on first loop, minus minimal stub (3AA)
    ext_length = int(np.linalg.norm(end_points[1]-end_points[0])/1.51)-4
    
    #diversify helix length
    lMod = list(range(-length_mod,length_mod+1))
    
    #indexList record helix length then loop number, 
    indexList = np.ones((len(lMod)))*ext_length+lMod 
    indexList=indexList.astype(np.int32)
    indexList[indexList<0] = 0 #remove negative lengths
    h1=list(map(extend_helix,repeat(build.copy()),indexList))
    pyList = list(map(np.append, repeat(build.copy()) ,h1 , repeat(0)))
    buildList = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        buildList[i] = c
    
    
    #align endpoints helix 1 to z-axis [aligns with stub build]
    ep_1 = return_aligned(end_points.copy())

    #convert epIn to match helix length exactly, one epGuide per helix length
    #epGuide = np.broadcast_to(ep_1,(iL_h1.shape[0], ep_1.shape[0],ep_1.shape[1])).copy()
    epGuide = np.repeat(np.expand_dims(ep_1,axis=0),indexList.shape[0],axis=0)

    tmp=np.array([0,0,-1.51,0])*np.reshape(indexList+4,(-1,1)) #1.51 is rise per res of ideal helix
    #assign ideal helix lengths to the guide (+4) single stub on first loop
    #starting stub is not added here since it ends at 0,0,0
    tmp[:,3]=1 # set rotation helper to 1
    epGuide[:,1,:] = tmp # reassign first helix endpoint #2
    
    return buildList, indexList, epGuide
    
    


# In[7]:


def first_loop(buildList, indexList, epGuide, neighbors=5, phiQueryNum=10, randMult=0,distCut=6):
    
    axisRot = np.array([0,0,-1,1]) 
    
    #query kdTree for the right loops, rotated tree large with overlaps -----------------------------
    query_ep = np.array(list(map(get_query,epGuide,repeat(0)))) # get query from endpoints 
    phiQ = np.linspace(-1,1,num=phiQueryNum) #generate phi query numbers
    #expand query to accomodate all phi query combinations
    queBroad = np.array(list(starmap(np.broadcast_to,zip(query_ep,repeat((phiQueryNum,query_ep.shape[1]))))))
    pQ = np.broadcast_to(phiQ,(queBroad.shape[:-1])) #expand to match endpoints query size
    phiQ_attach = np.expand_dims(pQ,axis=len(pQ.shape)) #expand dimensions to concatenate
    
    tree_query = np.concatenate((queBroad,phiQ_attach),axis=2) # concatenate phi for query
    tree_query = tree_query.reshape((-1,7))
    
    #original query code
    if randMult < 2:
        mapfunc = partial(binTreePhi.query, k=neighbors)
        large_answer = np.array(list(map(mapfunc, tree_query.reshape((-1,7)) ))) #Query the kD 
        large_answer1 = np.array(large_answer[:,1],dtype=np.int32) # get only the loop indices in position 1
        #awkward nest map to convert index of large_answer and remove repeated rotated indices
        answer = np.fromiter(map(convert_index,large_answer1.ravel()),dtype=np.int32)
    else:
        #randMult code to get more diverse loops, possibly include again
        mapfunc = partial(binTreePhi.query, k=neighbors*randMult)
        large_answer = np.array(list(map(mapfunc, tree_query.reshape((-1,7)))))
        #get the first five and randomly pick the rest of the neighbors
        nearNeigh = 5 #closest neighbors to keep
        indexer = np.hstack((np.array(range(nearNeigh)),np.random.choice(range(nearNeigh,neighbors*randMult), neighbors-nearNeigh,replace=False)))
        large_answer2 = np.array(large_answer[:,1,indexer],dtype=np.int32)
        #convert index of large_answer (rotated) to singular one index one loop (small_tree)
        answer = np.fromiter(map(convert_index,large_answer2.ravel()),dtype=np.int32)
        
    
    #expand epGuide and create epTrue to keep track of actual endpoints assembled
    epGuide = np.repeat(epGuide,neighbors*phiQueryNum,axis=0)
    epTrue = np.zeros((epGuide.shape))
    
    
    #record loops to try with helix lenghts to prevent repeats
    iL = np.repeat(indexList,neighbors*phiQueryNum) #expand helix length to match loop neighbors and phi expansion
    phiList = np.repeat(phiQ_attach.ravel(),neighbors) #expand phiList to match neighbors expansion
    iL = np.hstack((iL.reshape((-1,1)),answer.reshape((-1,1)))) #create index list to prevent repeated 

    #check the index list for repeated entrys and remove
    iL, u_indices =np.unique(iL,axis=0,return_index=True)
    
    phiList = phiList[u_indices] #remove repeated phi queries
    epGuide = epGuide[u_indices]
    epTrue = epTrue[u_indices]
    answer = answer[u_indices]
    
    #ep guide needs to be updated to start with the helix phi that matches the loop
    epGuide = np.array(list(map(rotate_ep,epGuide,answer,iL[:,0]))) #rotate guide to match
    epGuide = np.round(epGuide,6)
    epTrue[:,:2,:] = epGuide[:,:2,:] #copy ideal helix length from epGuide
    
    #update epTrue with the actual loop endpoints built
    feat_True = feats_acc[answer]
    epTrue[:,2,:3] = feat_True[:,:3] + epTrue[:,1,:3]
    epTrue[:,2,3] = 1
    #align to epTrue to initial rotation, the loop library was referenced on a 10helix expansion from 3AA stub
    offset = (iL[:,0]-10)*1.74533 #100deg in radians for 3.6 residues per turn
    xform_True = np.array(list(map(nu.xform_from_axis_angle_rad,repeat(axisRot),offset)))
    epTrue = np.array(list(map(nu.xform_npose,xform_True,epTrue)))
    
    
    #calculate deviation from guide endpoints to actual build
    dist = np.max(np.linalg.norm(epTrue[:,:3,:3]-epGuide[:,:3,:3],axis=2),axis=1)
    #remove distances further than the cut off
    disCutIndex = dist<distCut
    
    if not forgeAhead(disCutIndex,iL.shape[0]):
        iL = iL[np.zeros(iL.shape,dtype=np.bool8)]
        return iL, iL, iL, iL, iL , iL, iL

    #remove distance cut off 
    epGuide = epGuide[disCutIndex]
    epTrue = epTrue[disCutIndex]
    phiList = phiList[disCutIndex]
    iL = iL[disCutIndex]
    feat_True = feat_True[disCutIndex]
    xform_True = xform_True[disCutIndex]
    
    #expand build list to match indexList helix lengths
    #get the indices for each helix length segment to correspond with each build
    helixLengths = np.unique(iL[:,0])
    helix_indices = np.array(list(map(lambda x: iL[:,0]==x,helixLengths)))
    helix_number  = list(map(np.sum,helix_indices))

    #expand build lists so that is one for each loop to be added
    a = map(lambda x,y : np.repeat(np.expand_dims(x,axis=0),y,axis=0), buildList, helix_number)
    bL3 = []
    for x in a:
        bL3.extend(x.astype(np.float64))
        
    
    #loop alignment and
    #align loop and append to end of build
    hLoop = list(starmap(align_loop,zip(bL3,all_loops[iL[:,1]])))
    
    #append loop to builds, do not need clash checks since loops should clash with their own helix
    pyList = list(map(np.append, bL3 , hLoop , repeat(0))) #append aligned loop to build
    bL = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        bL[i] = c
    
    return bL, iL, epGuide, epTrue, phiList, feat_True, xform_True

        
    


# In[8]:


def second_helix(buildList, indexList, epGuide, epTrue, phiList, hnum, prev_loopFeature, prev_loopTrans, length_mod=1, distCut=6):

    size = epGuide.shape[-2] #to determine when the terminal helices occurs
    ind_hnum = hnum*2 # first helix, hnum 0, [0,1] indices for endpoints

    #controls extension length, loops have 4AA stubs of helices on each end. 
    #account for terminal helices -4, -8 for interior (2 loops)
    account = 8

    #extend helix length
    ext_length =  np.fromiter(map(np.linalg.norm,(epGuide[:,ind_hnum+1,:3] - epTrue[:,ind_hnum,:3])/1.51-account),dtype=np.int32)

    #diversify helix length
    lMod = np.array(list(range(-length_mod,length_mod+1)))
    new_helix_length = np.repeat(ext_length,len(lMod)) + np.repeat([lMod],ext_length.shape[0],axis=0).reshape(-1)
    
    #expand lists to accomdate new helix lenghts
    indexList = np.repeat(indexList,len(lMod),axis=0)
    epGuide = np.repeat(epGuide,len(lMod),axis=0)
    epTrue = np.repeat(epTrue,len(lMod),axis=0)
    phiList = np.repeat(phiList,len(lMod))
    buildList = np.repeat(buildList,len(lMod),axis=0)

    feat_True = np.repeat(prev_loopFeature,len(lMod),axis=0)
    xform_True = np.repeat(prev_loopTrans,len(lMod),axis=0)
    
    #add new helix lenghts to index LIst
    indexList = np.hstack((indexList,new_helix_length.reshape((-1,1))))
    
    #update true vector
    next_true_endpoint = np.hstack((feat_True[:,3:-1],np.ones((feat_True.shape[0],1))))
    rotVec = np.array(list(map(nu.xform_npose,xform_True,next_true_endpoint)))
    
    #quickMod
    epTrue[:,ind_hnum+1,:3] = epTrue[:,ind_hnum,:3] + rotVec[:,0,:3]*((indexList[:,ind_hnum]+account)*1.51).reshape((-1,1)) #account doublestub
    epTrue[:,:ind_hnum+2,3] = 1 #set rotation helper to 1
    
    #check distance deviation of next endpoints
    dist = np.linalg.norm(epTrue[:,ind_hnum+1,:3]-epGuide[:,ind_hnum+1,:3], axis=1)
    disCutIndex = dist<distCut
    
    if not forgeAhead(disCutIndex,buildList.shape[0]):
        iL = indexList
        iL = iL[np.zeros(iL.shape,dtype=np.bool8)]
        return iL, iL, iL, iL, iL
    
    #remove distance cut off 
    epGuide = epGuide[disCutIndex]
    epTrue = epTrue[disCutIndex]
    phiList = phiList[disCutIndex]
    indexList = indexList[disCutIndex]
    buildList = buildList[disCutIndex]
    
    #append hnext
    pyList = list(map(extend_helix,buildList,indexList[:,ind_hnum]))
    hnext = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        hnext[i] = c

    clashCheck = np.invert(np.fromiter(map(check_clash,buildList,hnext),dtype=np.bool8))
    if not forgeAhead(clashCheck,buildList.shape[0]):
        iL = indexList
        iL = iL[np.zeros(iL.shape,dtype=np.bool8)]
        return iL, iL, iL, iL, iL

    #remove distance cut off 
    epGuide = epGuide[clashCheck]
    epTrue = epTrue[clashCheck]
    phiList = phiList[clashCheck]
    indexList = indexList[clashCheck]
    buildList = buildList[clashCheck]
    hnext = hnext[clashCheck]
    
    pyList = list(map(np.append, buildList ,hnext , repeat(0)))
    buildList2 = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        buildList2[i] = c
          
    #prep phiList for hstack later
    phiList = phiList.reshape((-1,1))
    
    return buildList2, indexList, epGuide, epTrue, phiList
    
    
    


# In[9]:


#seems good up to second helix
#check

def next_loop_helix(buildList, indexList, epGuide, epTrue, phiList, hnum, 
                    neighbors=5, phiQueryNum=10, randMult=10, distCut=6, length_mod=1):
    
    ind_hnum = hnum*2
    
    #---------------Add Loop--------------------
    #apply approriate reference frame for the end of the build, query loop library
    xf = np.array(list(map(get_transform,repeat(ref),buildList))) #align the end of build to the reference helix to queary
    xf_rev = np.array(list(map(np.linalg.inv,xf)))  #reverse rotation
    epGuide_ref = np.array(list(map(nu.xform_npose,xf,epGuide))) #rotate endpoints to get query based on reference
    epTrue_ref = np.round(np.array(list(map(nu.xform_npose,xf,epTrue))),6)
    
    #get query from endpoints
    query_ep = np.array(list(map(get_query_true,epGuide_ref,epTrue_ref,repeat(hnum))))# 1 indexes second loop
    
    phiQ = np.linspace(-1,1,num=phiQueryNum)
    
    #expand query to accomodate all phi query combinations
    queBroad = np.array(list(map(np.broadcast_to,query_ep,repeat((phiQueryNum,query_ep.shape[1])))))
    pQ = np.broadcast_to(phiQ,(queBroad.shape[:-1])) #expand to match endpoints query size
    phiQ_attach = np.expand_dims(pQ,axis=len(pQ.shape)) #expand dimensions to concatenate
    
    tree_query = np.concatenate((queBroad,phiQ_attach),axis=2) # concatenate phi for query
    tree_query = tree_query.reshape((-1,7))
    
    #original query code
    if randMult < 2:
        mapfunc = partial(binTreePhiS.query, k=neighbors)
        answer = np.array(list(map(mapfunc, tree_query.reshape((-1,7)) ))) #Query the kD 
        answer = np.array(answer[:,1],dtype=np.int32) # get only the loop indices in position 1
        #awkward nest map to convert index of large_answer and remove repeated rotated indices
        answer = np.fromiter(map(convert_index, answer.ravel()),dtype=np.int32)
    else:
        #randMult code to get more diverse loops, possibly include again
        mapfunc = partial(binTreePhiS.query, k=neighbors*randMult)
        answer = np.array(list(map(mapfunc, tree_query.reshape((-1,7)))))
        #get the first five and randomly pick the rest of the neighbors
        nearNeigh = 5 #closest neighbors to keep
        indexer = np.hstack((np.array(range(nearNeigh)),np.random.choice(range(nearNeigh,neighbors*randMult), neighbors-nearNeigh,replace=False)))
        answer = np.array(answer[:,1,indexer],dtype=np.int32)
        #convert index of large_answer (rotated) to singular one index one loop (small_tree)
        answer = np.fromiter(map(convert_index,answer.ravel()),dtype=np.int32)
        
    
    #expand epGuide and create epTrue to keep track of actual endpoints assembled
    epGuide = np.repeat(epGuide,neighbors*phiQueryNum,axis=0)
    epTrue = np.repeat(epTrue,neighbors*phiQueryNum,axis=0)
    phiList =  np.repeat(phiList,neighbors*phiQueryNum,axis=0)
    xf = np.repeat(xf,neighbors*phiQueryNum,axis=0)
    xf_rev = np.repeat(xf_rev,neighbors*phiQueryNum,axis=0)
    buildList = np.repeat(buildList,neighbors*phiQueryNum,axis=0)

    epTrue_ref  = np.repeat(epTrue_ref,neighbors*phiQueryNum,axis=0)
    epGuide_ref = np.repeat(epGuide_ref,neighbors*phiQueryNum,axis=0)
    indexList = np.repeat(indexList,neighbors*phiQueryNum,axis=0)
    
    #record loops and helices in index list to prevent repeats
    indexList = np.hstack((indexList,answer.reshape((-1,1))))
    #record phi bins
    phiList = np.hstack((phiList,np.repeat(phiQ_attach.ravel(),neighbors).reshape((-1,1))))
    
    #check the index list for repeated entrys and remove
    indexList, u_indices =np.unique(indexList,axis=0,return_index=True)
    
    phiList = phiList[u_indices] #remove repeated phi queries
    epGuide = epGuide[u_indices]
    epGuide_ref = epGuide_ref[u_indices]
    answer = answer[u_indices]
    epTrue_ref = epTrue_ref[u_indices] 
    buildList = buildList[u_indices]
    xf_rev = xf_rev[u_indices]
    
    #update epTrue with the actual loop endpoints built
    true_vector = feats_acc[answer]
    epTrue_ref[:,ind_hnum+2,:3] = epTrue_ref[:,ind_hnum+1,:3] + true_vector[:,:3]
    epTrue_ref[:,:ind_hnum+3,3] = 1 #set rotation helper to 1
    
    #check distance deviations and remove
    dist = np.linalg.norm(epTrue_ref[:,ind_hnum+2,:3]-epGuide_ref[:,ind_hnum+2,:3], axis=1)
    disCutIndex = dist<distCut
    if not forgeAhead(disCutIndex, buildList.shape[0]):
        iL = indexList
        iL = iL[np.zeros(iL.shape,dtype=np.bool8)]
        return iL, iL, iL, iL, iL

    indexList = indexList[disCutIndex]
    phiList = phiList[disCutIndex] 
    epGuide = epGuide[disCutIndex]
    epGuide_ref = epGuide_ref[disCutIndex]
    epTrue_ref = epTrue_ref[disCutIndex]
    true_vector = true_vector[disCutIndex]
    buildList = buildList[disCutIndex]
    answer = answer[disCutIndex]
    xf_rev = xf_rev[disCutIndex]
    
    #append hnext
    pyList = list(map(align_loop,buildList,all_loops[answer]))
    hLoop_next = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        hLoop_next[i] = c
    
    
    clash_check = np.invert(np.fromiter(map(check_clash,buildList,hLoop_next),dtype=np.bool8))
    if not forgeAhead(clash_check, buildList.shape[0]):
        iL = indexList
        iL = iL[np.zeros(iL.shape,dtype=np.bool8)]
        return iL, iL, iL, iL, iL

    indexList = indexList[clash_check]
    phiList = phiList[clash_check] 
    epGuide = epGuide[clash_check]
    epGuide_ref = epGuide_ref[clash_check]
    epTrue_ref = epTrue_ref[clash_check]
    true_vector = true_vector[clash_check]
    buildList = buildList[clash_check]
    hLoop_next = hLoop_next[clash_check]
    xf_rev = xf_rev[clash_check]

    pyList = list(map(np.append, buildList ,hLoop_next , repeat(0)))
    buildList = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        buildList[i] = c
    
    
    #---------------Add Helix--------------------
    ind_hnum += 2 #iterate to next helix
    size = epGuide.shape[1] #number of endpoints
    #if we are at terminal helix, only subtract 4 for one loops worth of helical stub
    #+1 is used in code to index final endpoint, +1 for 0 array indexing, +1 for greater than, =+3
    if ind_hnum + 3 > size:
        account = 4
    else:
        account = 8

    #helical extension length
    ext_length = np.fromiter(map(np.linalg.norm,(epGuide_ref[:,ind_hnum+1,:3]-epTrue_ref[:,ind_hnum,:3])/1.51-account),dtype=np.int32)
    #diversity helix length
    lMod = np.array(list(range(-length_mod,length_mod+1)))
    new_helix_length = np.repeat(ext_length,len(lMod)) + np.repeat([lMod],ext_length.shape[0],axis=0).reshape(-1)
    
    #expand lists to accomodate new helix lenghts
    indexList = np.repeat(indexList,len(lMod),axis=0)
    epGuide = np.repeat(epGuide,len(lMod),axis=0)
    epTrue_ref = np.repeat(epTrue_ref,len(lMod),axis=0)
    phiList = np.repeat(phiList,len(lMod),axis=0)
    buildList = np.repeat(buildList,len(lMod),axis=0)
    true_vector = np.repeat(true_vector,len(lMod),axis=0)
    xf_rev = np.repeat(xf_rev,len(lMod),axis=0)

    #add new helix lengths to index LIst
    indexList = np.hstack((indexList,new_helix_length.reshape((-1,1))))
    
    #Update true build endpoints in reference frame and reverse back to original orientation
    next_ep_vector = true_vector[:,3:-1]
    epTrue_ref[:,ind_hnum+1,:3] = epTrue_ref[:,ind_hnum,:3]+next_ep_vector[:,:3]*((new_helix_length+account)*1.51).reshape((-1,1))
    epTrue_ref[:,:ind_hnum+2,3] = 1 #set rotation helper to 1
    epTrue = np.array(list(map(nu.xform_npose,xf_rev, epTrue_ref))) #reverse reference to match actual atom build
    
    #check distance for endpoint of helix extension and remove violations
    
    dist = np.linalg.norm(epTrue[:,ind_hnum+1,:3]-epGuide[:,ind_hnum+1,:3], axis=1)
    disCutIndex = dist<distCut
    if not forgeAhead(disCutIndex, buildList.shape[0]):
        iL = indexList
        iL = iL[np.zeros(iL.shape[0],dtype=np.bool8)]
        return iL, iL, iL, iL, iL

    indexList = indexList[disCutIndex]
    phiList = phiList[disCutIndex] 
    epGuide = epGuide[disCutIndex]
    buildList = buildList[disCutIndex]
    epTrue = epTrue[disCutIndex]
    
    #append hnext
    pyList = list(map(extend_helix,buildList,indexList[:,ind_hnum]))
    hnext = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        hnext[i] = c
    

    clashCheck = np.invert(np.fromiter(map(check_clash,buildList,hnext),dtype=np.bool8))
    if not forgeAhead(clashCheck, buildList.shape[0]):
        iL = indexList
        iL = iL[np.zeros(iL.shape[0],dtype=np.bool8)]
        return iL, iL, iL, iL, iL
    
    #remove clashes 
    epGuide = epGuide[clashCheck]
    epTrue = epTrue[clashCheck]
    phiList = phiList[clashCheck]
    indexList = indexList[clashCheck]
    buildList = buildList[clashCheck]
    hnext = hnext[clashCheck]
    
    #append helix to build
    pyList = list(map(np.append, buildList ,hnext , repeat(0)))
    buildList = np.empty((len(pyList),),dtype=np.object_)
    
    for i,c in enumerate(pyList):
        buildList[i] = c
        
    return buildList, indexList, epGuide, epTrue, phiList


# In[10]:


def test_ouput(bL, eT,helix_Num=0):
    nu.dump_npdb(bL[0],f'output/build_helix{helix_Num}.pdb')
    hf.HelicalProtein.makePointPDB(eT[0],f'et_helix{helix_Num}.pdb',outDirec='output/')


def add_loops(end_points, neighbors=10, length_mod=1, dist_cut=6, phiQueryNum=10, randMult=10, maxPhi_cut=20,
              uniquePhi=True, verbose=False):
    """Chains first_helix, first_loop, second helix and the iteratively loop_then_helix to add all loops to endpoints."""
    
    epIn = end_points.copy()
    size = int(len(epIn)/2) #number of helixes represented by helical endpoints
    
   
    bL, iL, epGuide = first_helix(epIn,length_mod=length_mod)
    if verbose:
        print(f'first helix: #{iL.shape[0]}')
    
    if iL.shape[0] == 0:
        if verbose:
            print('fail')
        return bL, iL, phiList, iL, iL, False
    
    bL, iL, epGuide, epTrue, phiList, loopFeature, xform_True = first_loop(bL, iL, epGuide, 
                                                                neighbors=neighbors, phiQueryNum=phiQueryNum,
                                                                           randMult=randMult, distCut=dist_cut)
    if verbose:
        print(f'first loop: #{iL.shape[0]}')
    if iL.shape[0] == 0:
        if verbose:
            print('fail')
        return bL, iL, phiList, iL, iL, False
    
    hnum=1
    bL, iL, epGuide, epTrue, phiList = second_helix(bL, iL, epGuide, epTrue, phiList, hnum, loopFeature, 
                                                   xform_True, length_mod=length_mod, distCut=dist_cut)
    if verbose:
        print(f'second helix: #{iL.shape[0]}')
    
    if iL.shape[0] == 0:
        if verbose:
            print('fail')
        return bL, iL, phiList, iL, iL, False
    
    for x in range(hnum,size-1):
        bL, iL, epGuide, epTrue, phiList = next_loop_helix(bL, iL, epGuide, epTrue, phiList, x, 
                neighbors=neighbors, phiQueryNum=phiQueryNum, randMult=randMult, distCut=dist_cut,length_mod=length_mod)
        
        if iL.shape[0] == 0:
            if verbose:
                print('fail2')
            return bL, iL, phiList, iL, iL, False
        elif iL.shape[0] > 40:
            indexer = random_reduce(bL,num_to_keep = maxPhi_cut)
            bL = bL[indexer]
            iL = iL[indexer] 
            epGuide = epGuide[indexer]
            epTrue =  epTrue[indexer]
            phiList = phiList[indexer]
            if verbose:
                print(f'{x+2} helix: reduced to #{maxPhi_cut}')
        if verbose:
            print(f'{x+2} helix: #{iL.shape[0]}')
         
    #only return proteins with a unique set of loops queried from different phi bins
    if uniquePhi:
        phiList, u_indices = np.unique(phiList,axis=0,return_index=True)
        epGuide = epGuide[u_indices]
        epTrue = epTrue[u_indices]
        iL = iL[u_indices]
        bL = bL[u_indices]
    
    return bL, iL, phiList, epGuide, epTrue, True
    
    
    

def add_loops_cycle(endpoints_in, numCycles=5, neighInc=5, lmodInc=1,phiQueryNum=10, randMult=0,
                    neighStart = 5, lengthStart=0, dist_cut=6,outDirec='output/', analysisOnly=True,printStats=True):
    
    start = time.time()
    total_structures = 0
    ePSuc = []
    lmod_max = 3
    
    endpoints = endpoints_in.copy()
    ep_index = np.array(range(len(endpoints)),dtype=np.int32)
    ep_index_cur = ep_index.copy()
    
    
    
    
    neigh_increase = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    lmod_increase =  np.array([0, 1, 2, 2, 2, 3, 3, 3])
    cycles = np.hstack((lmod_increase.reshape((-1,1)),neigh_increase.reshape((-1,1))))
    
    
    
    for i,alpha in enumerate(cycles):
        if printStats:
            print(f' ')
            print(f' ')
            print(f'cycle: {i}')
            
        for x in ep_index_cur:
            bL, iL, phiList, epGuide, epTrue, success = add_loops(endpoints[x],dist_cut=dist_cut,
                                                                  neighbors=neighStart + alpha[1]*neighInc,
                                                                  length_mod=lengthStart + alpha[0]*lmodInc, randMult=randMult, 
                                                                  phiQueryNum=phiQueryNum, verbose=printStats)
            if not success:
                if len(bL)>0:
                    return bL
                else:
                    continue
            
            tot=bL.shape[0]
            
            ePSuc.append(x)
            total_structures += tot
            if printStats:
                print(f'Num_Structs{x}x:   ', tot)
            
            if  not analysisOnly:
                for num,prot in enumerate(bL):
                    nu.dump_npdb(prot,f'{outDirec}build{x}_{num}.pdb')
                    
                    
        ep_index_cur = np.delete(ep_index,np.array(ePSuc,dtype=np.int32))
        if printStats:
            print('\n')
        
                
    end = time.time()    
    if printStats:
        print(f'Elapsed time: {end - start:.2f}')
        print(f'Unique Structures: {len(ePSuc)}   Unique Phis: {total_structures}')
        if(total_structures>0):
            print(f'{(end-start)/total_structures:.2f}s per structure')
            
            
    
    return ePSuc, total_structures


def bb_analyze(name,batch=32,z=12,loopTry=True,print_output=True,analysisOnly=True, outDirec=''):
    """Gets backbones stats recon error, clashes, precent core and looped success."""
    
    labels = ['name','batch','time','MSE Recon', 'No Clash', 'Mean Clash', 'Percent Core','Looped']
    
    brec = ge.BatchRecon(name=name)

    start = time.time()

    brec.generate(z,batch_size=batch)
    brec.MDS_reconstruct_()
    brec.to_npose()
    end = time.time()

    cc =  []
    scn_core = []

    for x in range(len(brec.npose_list)):
        cc.append(whole_prot_clash_check(brec.npose_list[x],brec.helixLength_list[x]))
        neighs = get_neighbor_2D(brec.npose_list[x])
        scn_core.append(get_scn(neighs))
    
    cc = np.array(cc)
    
    print(f'Structures Generation Attempts: {batch}')
    print(f'MSE for recon is {brec.reconstructionError():.2f} Angstroms')
    print(f'Elapsed time: {end - start:.2f}')
    print(f'{(end-start)/batch:.2f}s per structure')
    print(f'No Clash Structures: {len(np.where(cc<1)[0])}')
    print(f'Two Atoms or less Clash Structures: {len(np.where(cc<3)[0])}')
    print(f'Clashed Atoms Mean: {np.mean(cc):.2f} +/- {np.std(cc):.2f}')
    print(f'Percent Core: {np.mean(scn_core):.2f} +/- {np.std(scn_core):.2f}')
    
    if loopTry:
        start = time.time()
        ep1, tot = add_loops_cycle(np.array(brec.reconsMDS_),lengthStart=0, printStats=print_output,
                                   analysisOnly=analysisOnly, outDirec=outDirec)
        end = time.time()
        print(f'Loop Success: {len(ep1)}. Phi Reduced Structures: {tot}')
        print(f'Total time: {end-start:.2f}')
        print(f'Time Per Unique Topology: {(end-start)/len(ep1):.2f}s')
        print(f'Time Per Phi Bins Struct: {(end-start)/tot:.2f}s')
        
def bb_loop(name,batch=32,z=12,loopTry=True,print_output=True,analysisOnly=False, outDirec='', maxTry=32):
    """Gets backbones stats recon error, clashes, precent core and looped success for loaded endpoints."""
    
    epObj = ge.EP_Recon(name)
    epObj.to_npose()
    
    cc =  []
    scn_core = []

    
    for x in range(len(epObj.npose_list)):
        cc.append(whole_prot_clash_check(epObj.npose_list[x],epObj.helixLength_list[x]))
        neighs = get_neighbor_2D(epObj.npose_list[x])
        scn_core.append(get_scn(neighs))
    
    cc = np.array(cc)

    
    print(f'Num Endpoints: {len(epObj.npose_list)}')
    print(f'No Clash Structures: {len(np.where(cc<1)[0])}')
    print(f'2 or less atoms clashes: {len(np.where(cc<3)[0])}')
    print(f'Clashed Atoms Mean: {np.mean(cc):.2f}')
    print(f'Percent Core: {np.mean(scn_core):.2f}')
    
    if loopTry:
        start = time.time()
        ep1, tot = add_loops_cycle(np.array(epObj.endpoints_list),lengthStart=0,analysisOnly=analysisOnly,
                                   printStats=print_output, outDirec=outDirec)
        end = time.time()
    print(f'Loop Success: {len(ep1)}. Total Phi Bins Generated: {tot}')
    print(f'Total time: {end-start:.2f}')
    print(f'Time Per Unique Topology: {(end-start)/len(ep1):.2f}s')
    print(f'Time Per Phi Bins Struct: {(end-start)/tot:.2f}s')
        
        
        
        

if __name__ == "__main__":
    #
    
    parser = argparse.ArgumentParser(description="Lots more possible options to optimize loop try parameters.")
    
    
    parser.add_argument("-b","--batch", help="Number of Endpoints to Generate: Default 32",  default=32, type=int)
    parser.add_argument("-o", "--outdirec", help="Output Directory. Needs / :Default output/", default="output/")
    parser.add_argument("-i", "--infile", help="Location of generator network, No File extension. MinMax name needs to be of form {name}_mm", default="data/FullSet")
    parser.add_argument("-a", "--analyze_only", help="Just Display Analysis. Do not save pdb files",action="store_true")
    parser.add_argument("-e", "--endpoints", help="Input Data is numpy list of endpoints. Requires -i", action="store_true")
    parser.add_argument("-z", "--genInputSize", help="Size of vector to input to generator",default=12, type=int)
    parser.add_argument("-v", "--verbose", help="Display progress", action="store_true")
    args = parser.parse_args()

    if args.endpoints:
        if args.infile == "data/BestGenerator":
            print('Please give a numpy saved endpoint list using -i, no file extension')
        else:
            bb_loop(args.infile,batch=args.batch,z=args.genInputSize,analysisOnly=args.analyze_only,
               outDirec=args.outdirec, print_output=args.verbose)
    else:
        bb_analyze(args.infile,batch=args.batch,z=args.genInputSize,analysisOnly=args.analyze_only,
               outDirec=args.outdirec, print_output=args.verbose)
    
    
    
    
    


