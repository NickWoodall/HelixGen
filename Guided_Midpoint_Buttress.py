import tensorflow as tf
import os
import sys
#clean up these imports for unused later
from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,atan2,copysign
import numpy as np

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


import joblib
from sklearn.manifold import MDS
import argparse
from functools import partial
from itertools import starmap,repeat,permutations

from pymol import cmd, stored, selector

import GenerateEndpoints as ge
import HelixFit as hf
import FitTransform as ft

import seaborn as sns
import util.RotationMethods as rm
    
    
    
tf.config.run_functions_eagerly(True)


#load distance maps and endpoints dataset for initializing start
def load_distance_map(name, dm_file='data/Fits_4H_dm_phi.npz'):
    rr = np.load(dm_file, allow_pickle=True)
    X_train, y_train , featNames = [rr[f] for f in rr.files]
    
    
    return X_train[y_train==name][:,:-4]

def index_helix_ep(ep_in,helices_desired=[0,1],num_helices=4):
    
    num_ep = num_helices*2
    hi = np.array(helices_desired,dtype=int)
    h_ep = np.array(range(num_ep)).reshape((-1,2)) #generate helix to endpoint mapping
    
    #alternate example for indexing batch of X 
    #X.reshape((X.shape[0],-1))[:,indexarray]
    
    #select desired endpoints from  batch of endpoints
    return ep_in[np.ix_(np.array(range(ep_in.shape[0])),h_ep[hi].flatten(), np.array(range(ep_in.shape[2])))]
    
def get_midpoint(ep_in,helices_desired=[0,1],num_helices=4):
    
    num_ep = num_helices*2
    
    ind_ep = index_helix_ep(ep_in, helices_desired=helices_desired, num_helices=4)
    
    #calculate midpoint
    midpoint = ind_ep.sum(axis=1)/np.repeat(ind_ep.shape[1], ind_ep.shape[2])
    
    return midpoint

def get_stubs_from_points(ep_in,index=[0,1,2]):
#def get_stubs_from_n_ca_c(n, ca, c):
    """Modified from Brian's npose code  get_stubs_from_n_ca_c, index references 3 points, to define plane.
    """
    e1 = ep_in[:,index[1]]-ep_in[:,index[0]]
    e1 = np.divide( e1, np.linalg.norm(e1, axis=1)[..., None] )

    e3 = np.cross( e1, ep_in[:,index[2]]-ep_in[:,index[0]], axis=1 )
    e3 = np.divide( e3, np.linalg.norm(e3, axis=1)[..., None] )

    e2 = np.cross( e3, e1, axis=1 )

    stub = np.zeros((len(ep_in), 4, 4))
    stub[...,:3,0] = e1
    stub[...,:3,1] = e2
    stub[...,:3,2] = e3
    stub[...,:3,3] = ep_in[:,index[1]]
    stub[...,3,3] = 1.0

    return stub

def xform_npose_2batch(xform, npose):
    #single batch code  util.npose_util as xform_npose
    return np.matmul(np.repeat(xform[:,np.newaxis,...],npose.shape[1],axis=1),npose[...,None]).squeeze(-1)

def xform_to_z_plane(mobile, index_mobile=[0,1,2]):
    """rotate points into the z-plane for trilaterization. needs additional translation/reflection"""

    mobile_stub = get_stubs_from_points(mobile, index=index_mobile)
    mobile_stub_inv = np.linalg.inv(mobile_stub)
    
    z_plane_ref = np.repeat(np.array([[[0,0,0],[1,0,0],[1,1,0]]]), mobile.shape[0],axis=0)

    ref_stub = get_stubs_from_points(z_plane_ref, index=[0,1,2])

    xform = ref_stub @ mobile_stub_inv

    return xform


def rotate_base_tri_Zplane(endpoint_midpoints, target_point=4, index_mobile=[1,2,3], returnRotMat=False):
    """rotate points into the z-plane for trilaterization. Target point ensures that point is positive in Z"""
    tp = target_point #target point
    zplanexform = xform_to_z_plane(endpoint_midpoints,index_mobile=index_mobile) #one index start base triangle, default
    #add one for npose rot calc
    npose = np.concatenate((endpoint_midpoints, np.ones((endpoint_midpoints.shape[0],
                                                         endpoint_midpoints.shape[1],1))),axis=2) 
    rot = xform_npose_2batch(zplanexform,npose) # double batch matrix multiplication, see npose, for one batch

    #translate X domain to place first index of "index_mobile" to 0,0,0
    rot[:,:,0] = rot[:,:,0]-np.expand_dims(rot[:,index_mobile[0],0],axis=1)
    #based on target point guaranteed to be positive
    #reflect new points across the z axis to positive if negative to match just choosing positive solutions
    rot[...,2][rot[:,tp,2]<0] = -rot[...,2][rot[:,tp,2]<0]
    
    if not returnRotMat:
        return rot[...,:3] #remove npose rotate dimension
    else:
        return rot[...,:3], zplanexform



#-----------------methods to index needed indices from generator------------------------

def helix_dindex(helices_to_keep, num_helices=4, intraHelixDist=True):
    """Get index values for parts of the distance map"""
    
    #prep indices for distance map
    num_ep = num_helices*2
    mat_ind = np.array(range((num_ep)**2)).reshape((num_ep,num_ep))
    iu1 = np.triu_indices(num_ep, 1)
    
    helix_used = np.array(helices_to_keep,dtype=int)
    
    h_ep = np.array(range(num_ep)).reshape((-1,2)) #generate helix to endpoint mapping
    
    tot_ind = []
    
    if intraHelixDist:
        #get indices of distance map that correspond to each helix, overlap is distances between specified endpoints
        for x in helix_used:
            new_ind = np.intersect1d(mat_ind[h_ep[x]], mat_ind.T[h_ep[x]])
            tot_ind.extend(new_ind)
    
    
    for x in permutations(helix_used,2):
        new_ind = np.intersect1d(mat_ind[h_ep[x[0]]], mat_ind.T[h_ep[x[1]]])
        tot_ind.extend(new_ind)
    
    #convert to generator indices (indices of iu1 array)
    out_ind = []
    for x in tot_ind:
        if len(np.nonzero(mat_ind[iu1]==x)[0])>0:
            out_ind.append(np.nonzero(mat_ind[iu1]==x))

    return np.sort(np.array(out_ind).flatten())


def point_dindex(target_points, ref=[4], num_helices = 4):
    
    num_ep = num_helices*2
    mat_ind = np.array(range((num_ep)**2)).reshape((num_ep,num_ep))
    iu1 = np.triu_indices(num_ep, 1)
    
    dindex = []
    
    for tp in target_points:
        for ref_ind in ref:
            dindex.append(mat_ind[ref_ind,tp]) #indices for distances to target point
    
    dindex = np.array(dindex)
    
    out_ind = []
    for x in dindex.flatten():
        out_ind.append(np.nonzero(mat_ind[iu1]==x))
        
    out_ind = np.array(out_ind)
    
    return out_ind.reshape((len(target_points),-1))

def target_dindex(target_points, oneRef = True, num_helices = 5, baseTri_out=True):
    """Distance map indices for base triangle and output distance map"""
    
    num_ep = num_helices*2
    mat_ind = np.array(range((num_ep)**2)).reshape((num_ep,num_ep))
    iu1 = np.triu_indices(num_ep, 1)

    if oneRef:
        ref = [1,2,3]
        base_tri = [mat_ind[1][2],mat_ind[2][3],mat_ind[1][3]] #p1 to p2, p2 to p3, p1 to p3
        
    else:
        ref = [0,1,2]
        base_tri = [mat_ind[0][1],mat_ind[1][2],mat_ind[0][3]] #p0 to p1, p1 to p2, p0 to p3
    
    dindex = []
    
    for tp in target_points:
        dindex.append(mat_ind[ref,tp]) #indices for distances to target point
    
    dindex = np.array(dindex)
    
    out_ind = []
    for x in dindex.flatten():
        out_ind.append(np.nonzero(mat_ind[iu1]==x))
        
    out_ind = np.array(out_ind)
    
    return out_ind.reshape((-1,len(base_tri))),base_tri

def minMax_indices(distance_index, point_index, minmax_obj):
    
    #assemble conversions 
    #converts output from generator back to real distances
    dMin_all = tf.convert_to_tensor(minmax_obj.data_min_, dtype=tf.float32)
    mScale_all = tf.convert_to_tensor(minmax_obj.scale_, dtype = tf.float32)
    mMin = tf.convert_to_tensor(minmax_obj.feature_range[0], dtype = tf.float32)

    #index just the distances we need for calculation
    dMin = tf.gather(dMin_all, distance_index,axis=0)
    mScale = tf.gather(mScale_all, distance_index,axis=0)

    #indexes we need to determine the +/- z value of the new points
    pindex = point_dindex([5,6,7], ref=[4], num_helices = 4)
    dMin_nwp = tf.gather(dMin_all, point_index,axis=0)
    mScale_nwp = tf.gather(mScale_all, point_index,axis=0)
    
    return dMin, mScale, mMin, dMin_nwp,  mScale_nwp 

def ref_distmap_index(distances, num_helices = 4):
    
    num_ep = num_helices*2
    mat_ind = np.array(range((num_ep)**2)).reshape((num_ep,num_ep))
    iu1 = np.triu_indices(num_ep, 1)
    iu1_flat = iu1[0]*num_ep+iu1[1]
    
    return distances[np.ix_(range(distances.shape[0]),iu1_flat)]

def convert_dMat_to_iu1_index(indices_in, num_helices = 4):
    """Converts indices on flattened distance index to iu1 single indices"""
    
    
    conv_array = np.array(indices_in).flatten()
    
    num_ep = num_helices*2
    mat_ind = np.array(range((num_ep)**2)).reshape((num_ep,num_ep))
    iu1 = np.triu_indices(num_ep, 1)
    
    #convert to generator indices (indices of iu1 array)
    out_ind = []
    for x in conv_array:
        if len(np.nonzero(mat_ind[iu1]==x)[0])>0:
            out_ind.append(np.nonzero(mat_ind[iu1]==x))
            
    out_ind = np.array(out_ind)
        
    return out_ind.reshape(conv_array.shape)


# In[5]:


def prep_base_triangle_trilateriation(dindex, base_tri, distance_map):
    """Return x,y,z coords on z-plane of base triangle of tetrahedron from a distance map."""
    
#     dindex, base_tri = target_dindex(targ_dind, oneRef = oneRef, num_helices = num_helices)
# #     print(dindex)
# #     print(base_tri)
    
    #test case input data: prep base triangles for trilateration at zplane, (0,0,0) (dvar,0,0) (ivar,jvar,0)
    desired_dm = distance_map[:, base_tri] #base tri from dindex

    dvar_index = tf.convert_to_tensor(0 ,dtype=tf.int32)
    s2_index = tf.convert_to_tensor(2 ,dtype=tf.int32) # we would like the angle across from side 2
    s3_index = tf.convert_to_tensor(1 ,dtype=tf.int32)

    #x value representing center of 2nd sphere at (dvar,0,0) aka s1
    dvar = tf.reshape(tf.gather(desired_dm, dvar_index,axis=1),(-1,1)) #side 1
    s2 = tf.reshape(tf.gather(desired_dm,   s2_index,axis=1),(-1,1))
    s3 = tf.reshape(tf.gather(desired_dm,   s3_index,axis=1),(-1,1))

    #calculate the opposite angle of the the third side of base triangle using law of cosines
    s1sq = tf.square(dvar)
    s2sq = tf.square(s2)
    s3sq = tf.square(s3)
    ang3 = np.arccos((-s3sq+s2sq+s1sq)/(2*dvar*s2))

    #take third point of base triangle via distance * vector
    v13 = tf.concat([tf.cos(ang3), tf.sin(ang3), tf.zeros_like(ang3)], axis=1)
    p3 = s2*v13
    #center points of 3rd sphere
    ivar = tf.reshape(p3[:,0],(-1,1))
    jvar = tf.reshape(p3[:,1],(-1,1))


    #convert all to float32 to match generator output
    #expand to dindex size 

    dvar = tf.cast(tf.repeat(dvar,dindex.shape[0],axis=1),dtype=tf.float32)
    ivar = tf.cast(tf.repeat(ivar,dindex.shape[0],axis=1),dtype=tf.float32)
    jvar = tf.cast(tf.repeat(jvar,dindex.shape[0],axis=1),dtype=tf.float32)
    
    return dvar, ivar, jvar


#functions for back propagation


@tf.function 
def train_step(input_z_var,ref_map_, helix_keep_mask_,
               target_mp_, dvar_, ivar_, jvar_,
                dMin_, mScale_, mMin_, dMin_nwp_, mScale_nwp_, 
                dindex_, pindex_, batch_,z_reflect_ind_, scale_):

    with tf.GradientTape() as g_tape:
        g_tape.watch(input_z_var)
        g_o = gen_obj.g(input_z_var)
        masked_loss = maskLoss(ref_map_, g_o, helix_keep_mask_)
        mp_loss  = tf.divide(midpoints_loss(g_o, target_mp_, 
                    dvar_, ivar_, jvar_,
                    dMin_, mScale_, mMin_, dMin_nwp_, mScale_nwp_, 
                    dindex_, pindex_, batch_,z_reflect_ind_), scale_)

        loss = tf.reduce_sum(mp_loss,axis=1) + tf.reduce_sum(masked_loss,axis=1)

    g_grads = g_tape.gradient(loss, input_z_var)
    optimizer.apply_gradients(zip([g_grads],[input_z_var]))

    return input_z_var, masked_loss, mp_loss


@tf.function
def maskLoss(y_actual, y_pred,mask):
    """Loss Function for mantaing shape of input helices"""
    custom_loss_val = tf.multiply(mask,tf.square(y_actual-y_pred))
    return custom_loss_val

@tf.function
def midpoints_loss(g1, target, 
                   dvar, ivar, jvar,
                   dMin, mScale, mMin, dMin_nwp, mScale_nwp, 
                   dindex, pindex, batch_size, zr_ind):
    """Loss function to move output of two generated helices to target midpoint"""


    #now using dindex gather the desired indices for tetrahedron calcs

    #radius of the spheres, aka the distances to unmasked endpoints
    g2 = tf.gather(g1,dindex,axis=1)

    #see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
    #inspect .scale_
    conv_dist = tf.add(tf.divide(tf.subtract(g2, mMin), mScale),dMin)
    #transpose lets you easily grab all distances with gather/axis 
    conv_dist_squared = tf.transpose(tf.square(conv_dist),perm=[0, 2, 1]) 

    r1_sq = tf.gather(conv_dist_squared, 0, axis=1) 
    r2_sq =  tf.gather(conv_dist_squared,1, axis=1) 
    r3_sq = tf.gather(conv_dist_squared, 2, axis=1)

    #calculate coordinates of spherial intersect
    x = tf.divide(tf.add(tf.subtract(r1_sq,r2_sq),tf.square(dvar)),tf.multiply(2.0,dvar))
    y1 = tf.divide(tf.add(tf.add(tf.subtract(r1_sq,r3_sq), tf.square(ivar)), tf.square(jvar)),tf.multiply(2.0,jvar))
    y = tf.subtract(y1,tf.multiply(tf.divide(ivar,jvar),x))

    pre_z = tf.subtract(tf.subtract(r1_sq,tf.square(x)),tf.square(y))
    fixed_z = tf.clip_by_value(pre_z, 1e-10, 100)

    #adds  to negative values to 0 for sqrt,
    #I think is okay as zero z will imply lengthening of distances to match a non-zero target midpoint,
    #pushing the network in the desired direction?

    z = tf.sqrt(fixed_z) #assume positive solution
    z_neg = tf.multiply(z,-1) #assume negative soluation

    #new points, with both assumptions
    nwp = tf.concat((tf.reshape(x,(batch_size,-1,1)),
                    tf.reshape(y,(batch_size,-1,1)),
                    tf.reshape(z,(batch_size,-1,1))), axis=2)  #

    nwp_negz = tf.concat((tf.reshape(x,(batch_size,-1,1)),
                    tf.reshape(y,(batch_size,-1,1)),
                    tf.reshape(z_neg,(batch_size,-1,1))), axis=2)  #

    #some positive solutions assumptions,
    # assume first [i4] is actual positive use remaining distances of i4 to (i5,i6,i7) to determine z sign
    # closest to matching distance is used


    #let's start by calculating all i4 to (i5,i6,i7) distances

    #stop the gradients since these are used to index gather and scatter
    #unsqueeze at two different dimensionsq to broadcast into matrix MX1 by 1XN to MXN 
    nwp_p =  tf.stop_gradient(tf.expand_dims(nwp,axis=1) - tf.expand_dims(nwp,axis=2))
    nwp_n =  tf.stop_gradient(tf.expand_dims(nwp,axis=1) - tf.expand_dims(nwp_negz,axis=2))

    nwp_dist_pz = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(nwp_p), 3)),(-1,4,4)) #distance calc +1e6?
    nwp_dist_nz = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(nwp_n), 3)),(-1,4,4))  #distance calc

    z_pn_dist_pre_con = tf.gather(g1,pindex,axis=1)
    z_pn_dist = tf.add(tf.divide(tf.subtract(z_pn_dist_pre_con, mMin), mScale_nwp),dMin_nwp)

    #index p4 to p5,p6,p7
    #rewrite as non-slice version of this

    nwp_dist_pz_c = tf.squeeze(tf.gather(tf.gather(nwp_dist_pz, [0], axis=1), [1,2,3], axis=2))
    nwp_dist_nz_c = tf.squeeze(tf.gather(tf.gather(nwp_dist_nz, [0], axis=1), [1,2,3], axis=2))

    nwp_dist_pz_c = tf.expand_dims(nwp_dist_pz_c,axis=2)
    nwp_dist_nz_c = tf.expand_dims(nwp_dist_nz_c,axis=2)

    # #using a single distance decide the z assumption and apply
    correct_z_assum = tf.abs(z_pn_dist - nwp_dist_nz_c) < tf.abs(z_pn_dist - nwp_dist_pz_c)
    cz = tf.squeeze(tf.multiply(tf.cast(correct_z_assum,tf.int32),-2))

    z_reflect_tensor = tf.ones_like(nwp, dtype=tf.int32)

    nwp_mult = tf.cast(tf.tensor_scatter_nd_add(z_reflect_tensor, zr_ind, cz),dtype=tf.float32)
    nwp_final = tf.multiply(nwp_mult,nwp)

    midpoint = tf.reduce_mean(nwp_final,axis=1)
    return tf.square(tf.subtract(midpoint,target)) # means squared loss to desired midpoint


# In[23]:


def fullBUTT_GPU(gen_obj, ref_map, target_mp_in, batch_size=32,cycles=100, input_z=None, 
                          rate=0.05, target_ep=[4,5,6,7], num_helices=4, oneRef=True,
                          scale=5.0, z_size=12, print_loss=False):
    
    batch_indices = np.repeat(np.array(range(ref_map.shape[0])),batch_size)
    batch = batch_indices.shape[0]
    target_mp = tf.convert_to_tensor(np.repeat(target_mp_in, batch_size,axis=0),dtype=tf.float32)
    ref_map = np.repeat(ref_map, batch_size, axis=0)
    
    #establish indices for distances to reference
    #prep base triangle, convert distances from minmax to regular
    dindex, base_tri = target_dindex(target_ep, oneRef = True, num_helices = num_helices)
    base_tri = convert_dMat_to_iu1_index(base_tri) #dirty
    pindex = point_dindex(target_ep[1:], ref=[target_ep[0]], num_helices = num_helices)

    #convert generator output to 'real distances'
    #dMin, mScale, mMin, dMin_nwp,  mScale_nwp = minMax_indices(dindex, pindex, brec.mm)
    mmTuple = minMax_indices(dindex, pindex, gen_obj.mm)

    # prepare base triangle for trilateriation (z plane , p1 at origin, p2 positive x)
    #dvar, ivar, jvar = prep_base_triangle_trilateriation(dist[:batch], targ_dind = [4,5,6,7], oneRef = True, num_helices=4)
    #baseTuple = prep_base_triangle_trilateriation(dist[:batch], targ_dind = target_ep, oneRef = True, num_helices=num_helices)

    baseTuple = prep_base_triangle_trilateriation(dindex, base_tri, ref_map)
    
    dMin, mScale, mMin, dMin_nwp,  mScale_nwp = mmTuple
    dvar,ivar,jvar = baseTuple
    
    #mask for keeping buttress helices in same orientation
    h_index = helix_dindex([0,1], num_helices=4, intraHelixDist=True)
    helix_keep_mask = np.zeros((ref_map.shape[1],),dtype=np.int32)
    helix_keep_mask[h_index] = 1
    helix_keep_mask = tf.convert_to_tensor(helix_keep_mask,dtype=tf.float32)

    #input to generator (determinstic output)
    
    ref_map_conv = gen_obj.mm.transform(ref_map) #map to keep helices same as input
    #controlling z reflection during trilaterization
    z_r_innerInd = np.repeat(tf.convert_to_tensor([[[1,2],[2,2],[3,2]]]),batch,axis=0)
    #batch index
    zfi_bi =np.expand_dims(np.array(range(batch)).reshape((-1,1)).repeat(3,axis=1),axis=2)
    z_reflect_ind = np.concatenate((zfi_bi,z_r_innerInd),axis=2)
    
    
    if input_z is None:
        input_z = tf.random.uniform(shape=(batch, z_size), minval=-1, maxval=1)
        
    with tf.device(device_name):
        input_z_var = tf.Variable(input_z)
        ref_map_ = tf.constant(ref_map_conv,dtype=tf.float32)
        scale_, z_reflect_ind_ = tf.constant(scale), tf.constant(z_reflect_ind)
        target_mp_, batch_ = tf.constant(target_mp),  tf.constant(batch)
        dindex_, pindex_ = tf.constant(dindex), tf.constant(pindex)
        dMin_, mScale_, mMin_ = tf.constant(dMin), tf.constant(mScale), tf.constant(mMin),
        dMin_nwp_,  mScale_nwp_ =  tf.constant(dMin_nwp),  tf.constant(mScale_nwp)
        dvar_, ivar_, jvar_ = tf.constant(dvar), tf.constant(ivar), tf.constant(jvar)
        helix_keep_mask_ = tf.constant(helix_keep_mask, dtype=tf.float32)

    #store grads and inputs as we backpropagate
    z=[]
    loss_mask = []
    loss_mp = []
    grads = []
    

    if print_loss:
        g_o = gen_obj.g(input_z_var)
        masked_loss = maskLoss(ref_map_, g_o, helix_keep_mask_)

        mp_loss  = tf.divide(midpoints_loss(g_o, target_mp_, 
                            dvar_, ivar_, jvar_,
                            dMin_, mScale_, mMin_, dMin_nwp_, mScale_nwp_, 
                            dindex_, pindex_, batch_, z_reflect_ind_), scale_)

        print('start_masked',np.round(np.sum(masked_loss),2))
        print('start_mp',np.round(np.sum(mp_loss),2))
    
    for t in range(1,cycles):
        
        in_z, mask_l, mp_l = train_step(input_z_var,ref_map_, helix_keep_mask_,
                                        target_mp_, dvar_, ivar_, jvar_,
                                        dMin_, mScale_, mMin_, dMin_nwp_, mScale_nwp_, 
                                        dindex_, pindex_, batch_, z_reflect_ind_, scale_)
        
        z.append(in_z.numpy())
        loss_mask.append(mask_l.numpy())
        loss_mp.append(mp_l.numpy())

    if print_loss:
        print('end_masked', np.round(np.sum(loss_mask[-1]),2))
        print('end_mp', np.round(np.sum(loss_mp[-1]),2))
    
    return z, loss_mask, loss_mp, batch_indices
    
#give backpropagated generator output, find the best outputs based on loss
def buttress_ep_from_z(gen_obj, gen_z, starting_ep , loss_midpoint, loss_masked, batchIndices,
                       max_loss_mp = 0.001, max_loss_mask = 0.001):
    
    
    best_mp = np.sum(loss_midpoint<max_loss_mp,axis=1)>2 # 3 total mp loss outputs (x,y,z of midpoint to target)
    best_mask = np.sum(loss_masked<max_loss_mask,axis=1)>27 # 28 total mask loss point (2 helices)

    mask_mp_bool = np.logical_and(best_mp, best_mask)       

    identified_z = gen_z[mask_mp_bool]
    print(f'Outputs passing filters: {len(identified_z)}')
    print(f'Total Outputs: {len(gen_z)}')
    uInd = batchIndices[mask_mp_bool]
    
    orig_ep = starting_ep[uInd]
    
    
    gen_obj.generate(z=12, input_z = identified_z, batch_size=identified_z.shape[0])
    gen_obj.MDS_reconstruct_()
    
    out_ep = np.array(gen_obj.reconsMDS_)
    
    return align_generated_to_starting_ep(out_ep, orig_ep)

def buttress_ep_from_z_mask_only(gen_obj, gen_z,loss_masked, batchIndices,
                                 max_loss_mask = 0.002, max_out=100, print_stats= False):
    
    
    sm = np.sum(loss_masked,axis=1)
    smi = np.argsort(sm)
    sm_sort = sm[smi]
    best_mask = sm_sort < max_loss_mask
    
    ind2 = np.array(range(len(sm)))
    #uInd = batchIndices[smi][best_mask]
    uiInd = ind2[smi][best_mask]
    uInd = batchIndices[uiInd]
    
    if print_stats:
        print('Input Size: ',      len(sm))
        print('Passing Filters: ', len(uInd))
    
    
    if len(uInd)>max_out:
        uInd = uInd[:max_out]
        uiInd = uiInd[:max_out]
    
    identified_z = gen_z[uiInd]
    
    gen_obj.generate(z=12, input_z = identified_z, batch_size=identified_z.shape[0])
    gen_obj.MDS_reconstruct_()
    
    out_ep = np.array(gen_obj.reconsMDS_)
    
    return out_ep, uInd

def buttress_ep_from_z_mask_mp(gen_obj, gen_z,loss_masked, loss_mp_ed, batchIndices, max_mp_loss = 1e-3,
                                 max_loss_mask = 0.002, max_out=200, print_stats= False, mask_first=True):
    
    if mask_first:
        #make sure the input two helices are maintained in generator output
        sm = np.sum(loss_masked,axis=1)
        smi = np.argsort(sm) #get sorted indices of lowest mask loss
        sm_sort = sm[smi]
        smi_o = smi[sm_sort < max_loss_mask]
        smp = np.sum(loss_mp_ed, axis=1)

        ind2 = np.array(range(len(sm)))
        uInd = batchIndices[smi_o][smp[smi_o]<max_mp_loss]
        uiInd = ind2[smi_o][smp[smi_o]<max_mp_loss]
    else:
        #make sure the input two helices are maintained in generator output
        sm = np.sum(loss_mp_ed,axis=1)
        smi = np.argsort(sm) #get sorted indices of lowest mask loss
        sm_sort = sm[smi]
        smi_o = smi[sm_sort < max_mp_loss]
        smp = np.sum(loss_mp_ed, axis=1)

        ind2 = np.array(range(len(sm)))
        uInd = batchIndices[smi_o][smp[smi_o]<max_loss_mask]
        uiInd = ind2[smi_o][smp[smi_o]<max_loss_mask]
        

    if len(uInd) > max_out:
        uInd = uInd[:max_out]
        uiInd = uiInd[:max_out]
    
    if print_stats:
        print('Input Size: ',      len(sm))
        print('Passing Filters: ', len(uInd))
    
    
    
    identified_z = gen_z[uiInd]
    
    gen_obj.generate(z=12, input_z = identified_z, batch_size=identified_z.shape[0])
    gen_obj.MDS_reconstruct_()
    
    out_ep = np.array(gen_obj.reconsMDS_)
    
    return out_ep, uInd
    

#the output of generator will not perfectly match the desired input, get best fit via Kabsch
def align_generated_to_starting_ep(gen_ep, orig_ep, target_mp=None):
    """Uses Kabsh to align generated endpoints onto original endpoints. Orig_Ep on origin with oneRef."""
    #Thanks to below for this code; modified to batch form
    #moves gen_
    #https://pymolwiki.org/index.php/Kabsch

    #only center on first four points [first two helices]
    gen_ep_4 = gen_ep[:,:4,:].copy()
    orig_ep_4 = orig_ep[:,:4,:].copy()
    
    
    #centering to prevent affine transformaiton
    COM_orig = np.expand_dims(np.sum(orig_ep_4, axis=1)/orig_ep_4.shape[1]   ,axis=1)
    COM_gen =  np.expand_dims(np.sum(gen_ep_4,  axis=1)/gen_ep_4.shape[1]     ,axis=1)
    
    gen_ep_4_cen = gen_ep_4 - COM_gen
    orig_ep_4_cen = orig_ep_4 - COM_orig
    
    #initial error estimate
    #E0 = np.sum( np.sum(np.square(gen_ep_4),axis=1),axis=1) + np.sum( np.sum(np.square(orig_ep_4),axis=1),axis=1)

    # This beautiful step provides the answer.  V and Wt are the orthonormal
    # bases that when multiplied by each other give us the rotation matrix, U.
    # S, (Sigma, from SVD) provides us with the error!  Isn't SVD great!                                            #2                      #1
    V, S, Wt = np.linalg.svd( np.matmul(np.transpose(gen_ep_4_cen, axes=[0,2,1]), orig_ep_4_cen))

    # we already have our solution, in the results from SVD.
    # we just need to check for reflections and then produce
    # the rotation.  V and Wt are orthonormal, so their det's
    # are +/-1.
    reflect = np.linalg.det(V) * np.linalg.det(Wt)
    #original solution, I will take both reflections
    #multiples by 1 or -1 depending if relfect is negative (reflection)
    proper_reflection = ((reflect>0).astype(np.int32)*-2+1) 
    S[:,-1] = S[:,-1]*proper_reflection
    V[:,:,-1] = -V[:,:,-1]*proper_reflection.reshape((-1,1))

    V_reflect = V.copy()
    V_reflect[:,:,-1] = -V_reflect[:,:,-1]
    #Error
#     RMSD = E0 - (2.0 * np.sum(S,axis=1))
#     RMSD = np.sqrt(np.abs(RMSD / L))
    
    #generate rotation matrices
    U = np.matmul(V, Wt)
    U_reflect = np.matmul(V_reflect, Wt)

#    return target_mp,COM_gen,COM_orig
    if target_mp:
        final_target_midpoint = np.matmul(np.expand_dims((target_mp-COM_gen.squeeze(axis=1)),axis=1), U).squeeze() + COM_orig.squeeze(axis=1)
    final_ep_full         = np.matmul((gen_ep-COM_gen), U) + COM_orig
    final_ep_full_reflect = np.matmul((gen_ep-COM_gen), U_reflect) + COM_orig
    
    if target_mp:
        return final_ep_full, final_ep_full_reflect, final_target_midpoint
    else:
        return final_ep_full, final_ep_full_reflect

def guess_reflection(p, p_reflect, des_mp, invert=False):
    p_mp = get_midpoint(p, helices_desired=[2,3], num_helices=4)
    p_reflect_mp = get_midpoint(p_reflect, helices_desired=[2,3], num_helices=4)
    
    final_ep = np.zeros_like(p)

    measure1 = np.linalg.norm(p_mp - des_mp,axis=1)
    measure2 =  np.linalg.norm(p_reflect_mp - des_mp,axis=1)

    if not invert:
        final_ep[np.nonzero((measure1<measure2))] = p[np.nonzero((measure1<measure2))]
        final_ep[np.nonzero((measure1>measure2))] = p_reflect[np.nonzero((measure1>measure2))]
    else:
        final_ep[np.nonzero((measure1>measure2))] = p[np.nonzero((measure1>measure2))]
        final_ep[np.nonzero((measure1<measure2))] = p_reflect[np.nonzero((measure1<measure2))]
        
    
    return final_ep

def determine_reflection(input_points, input_points_reflect, reference_targets):
    
    i_mp = get_midpoint(input_points, helices_desired=[2,3]) 
    i_reflect_mp = get_midpoint(input_points_reflect, helices_desired=[2,3]) 



    #get a point on the desired line on mp dist away
    dmp = np.linalg.norm(reference_targets-np.expand_dims(i_mp,axis=1),axis=2)
    dmp_r = np.linalg.norm(reference_targets-np.expand_dims(i_reflect_mp,axis=1),axis=2)

    am = np.argmin(np.abs(dmp),axis=1)
    am_r = np.argmin(np.abs(dmp_r),axis=1)
    #desired reflection is the further along the target point array, highest index
    proper_reflect = am_r>am
    

    final = input_points.copy()
    final[proper_reflect] = input_points_reflect[proper_reflect]
    
    return final

def align_points_to_XYplane(input_points, keep_orig_trans=False):
    """Align batch of points to XY Plane."""
    
    #example data set
    #modified below to align set of points to the xy plane
    #https://www.mathworks.com/matlabcentral/answers/255998-how-do-i-move-a-set-of-x-y-z-points-so-they-align-with-the-x-y-plane
    
    COM_start  = np.sum(input_points, axis=1)/input_points.shape[1]
    input_points_ori = input_points - np.expand_dims(COM_start,axis=1)

    #sp_ori = (start_points - np.expand_dims(COM_start,axis=1))[0]
    V, S, Wt= np.linalg.svd(input_points_ori,full_matrices=False)
    
    #this indexing is opposite of the matlab output
    axis= np.cross(Wt[:,2,:],[[0,0,1]]);
    angle = -np.arctan2(np.linalg.norm(axis,axis=1), Wt[:,2,2])
    
    xf = []
    for x in range(len(axis)):
        xf.append(nu.xform_from_axis_angle_rad(axis[x], angle[x]))


    xformR = np.array(xf)
    R = xformR[:,0:3,0:3]
    if keep_orig_trans:
        A_rot = np.matmul(input_points_ori, R) + np.expand_dims(COM_start, axis=1)
    else:
        A_rot = np.matmul(input_points_ori, R) 
    
    return A_rot

def view_ep(ep_in, name='test', max_out=10):
    
    outDirec='output/'
    
    if len(ep_in)>max_out:
        it = max_out
    else:
        it = len(ep_in)
    
    for i in range(it):
        hf.HelicalProtein.makePointPDB(ep_in[i], f'{name}{i}.pdb', outDirec='output/')
        
    cmd.delete("all")
    for i in range(it):
        cmd.load(f'{outDirec}/{name}{i}.pdb')
        
    
    cmd.save(f'{outDirec}/viewEP_{name}.pse')
    
def vp(ep_in, guide_in,  name='test', max_out=10):
    
    outDirec='output/'
    
    if len(ep_in)>max_out:
        it = max_out
    else:
        it = len(ep_in)
    
    for i in range(it):
        hf.HelicalProtein.makePointPDB(ep_in[i], f'{name}{i}.pdb', outDirec='output/')
        hf.HelicalProtein.makePointPDB(ep_in[i], f'{name}{i}.pdb', outDirec='output/')
        
        
    hf.HelicalProtein.makePointPDB(guide_in,f'{name}_guide.pdb',outDirec='output/')
        
    cmd.delete("all")
    cmd.load(f'{outDirec}/{name}_guide.pdb')
    for i in range(it):
        cmd.load(f'{outDirec}/{name}{i}.pdb')
        
    
    cmd.save(f'{outDirec}/viewEP_{name}.pse')

#generate test set of points to align to a plane 
#qr decomposition generates a set of orthoganol vectors from the input matrix
#we can use this to matrix mulplication to offset a plane on the z axis
# Q,R=np.linalg.qr(np.random.randn(3,3))

# # [Q,~] = qr(randn(3));
# n = 500;
# sigma = 0.02;
# #grid of points
# gp = (np.random.rand(n,3)*[1, 1, 0] + sigma*np.random.randn(n,3))
# XYZ = gp@Q;
# X = XYZ[:,0];
# Y = XYZ[:,1];
# Z = XYZ[:,2];


def generate_parabola(start, stop, h=[20,20,20], k=[20,30,40], num_points = 100):
    #create a set of parabolas past through ([0,0]) start of helices to vertex ([[10,20],[10,30],[10,40]])
    # (x-h)^2 = -4(a)(y-k)
    # Solve for 'a' at x=0, y=0
    # (0-h)^2 = -4(a)(0-k)
    #      a  =  h^2/4k
    h = np.array(h).reshape((-1,1))
    k = np.array(k).reshape((-1,1))
    a = np.square(h)/(4*k)
    
    #trace x from 0 to 10
    # -4ay +4ak = (x-h)^2
    # -4ay = (x-h)^2 - 4ak
    #    y = ( (x-h)^2 - 4ak  ) / (-4a)
    #    y = (4ak - (x-h)^2)
    x = np.repeat(np.expand_dims(np.linspace(start,stop,num=100), axis=0), h.shape[0],axis=0)
    z = np.divide(4*a*k -np.square(x-h), (4*a))
   
    x=np.expand_dims(x,axis=2)
    z=np.expand_dims(z,axis=2)
    y = np.zeros_like(x)
    gen_para =  np.concatenate((x,y,z),axis = 2)
    
    return gen_para
    
#rotate parabola around the x axis
# angleDeg = 90
# gpi = 1
# xfr=nu.xform_from_axis_angle_deg([1,0,0],angleDeg)
# gp = np.hstack((g_para[gpi],np.ones_like(g_para[gpi,:,gpi].reshape((-1,1)))  ))
# gp_z=nu.xform_npose(xfr,gp)[:,:3]
# gp_z.shape

# plt.scatter(gp_z[:,0], gp_z[:,2], alpha=0.5)
# plt.axis('equal')
# plt.show()

def build_protein_on_guide(start_helices, guide_points, batch=200, 
                           next_mp_dist = 9,mp_deviation_limit = 5):
    
    #this orientation promotes most likely growth to [0,0,1] (see distribution of reference set)
    sh_xy = align_points_to_XYplane(start_helices, keep_orig_trans=False)
    ci = np.zeros(start_helices.shape[0],dtype=np.int32)
    master_ep = sh_xy[:,:4,...]# if there are more than 4ep (2 helices) just take the first two
    
    roomToGrow = True
    output_ep_list = []

    while roomToGrow:

        #second set of added points unused except for maintaining distance maps indexing from gen
        #(based around 4 helices)
        current_quad_prez = np.concatenate((master_ep[:,-4:,:], master_ep[:,-4:,:] ), axis=1)

        #get a point on the desired line on mp dist away
        mp_start = get_midpoint(current_quad_prez,helices_desired=[0,1])

        #guide_start = gp[ci]
        vg = np.repeat( np.expand_dims(guide_points, axis=0) , current_quad_prez.shape[0],axis=0)
        #make that move backwards much larger than next_mp_dist
        boo = np.repeat(np.expand_dims(np.arange(vg.shape[1]),axis=0), vg.shape[0],axis=0)
        boo2 = (boo<np.expand_dims(np.expand_dims(ci,axis=1),axis=1))[:,0]
        vg[boo2] = -1e6


        dmp = np.linalg.norm(vg - np.expand_dims(mp_start,axis=1),axis=2)
        am = np.argmin(np.abs(dmp - next_mp_dist),axis=1)
        print('max next indices',max(am))
        tmp = vg[np.ix_(np.array(range(vg.shape[0])), am, np.array(range(3)))][0]

        cqpz_dmp = np.concatenate((current_quad_prez, np.expand_dims(tmp,axis=1)), axis=1)
        print(current_quad_prez.shape)

        #rotate points and desired midpoint into trilaterization place
        current_quad_tmp = rotate_base_tri_Zplane(cqpz_dmp,  target_point=4, index_mobile=[1,2,3])

        target_midpoint = current_quad_tmp[:, 8, :]
        current_quad = current_quad_tmp[:, :8, :]
        #create distance map for generator
        start_dist = np.expand_dims(current_quad,axis=1) - np.expand_dims(current_quad,axis=2)
        dist = np.sqrt(np.sum(start_dist**2, 3))
        dist = dist.reshape((dist.shape[0],-1))

        #indices for reference map
        ref_map_base = ref_distmap_index(dist, num_helices=4)

        #GPU ##33s  with 500,000 samples with 200 cycles (average of 7 runs)
        #CPU ##39s
        #maybe there is something I can do to make this more effecient, not pipeline bottleneck so okay
        #for small models like this tensor flow says gpu may not be more effecient
        output_z, loss_mask, loss_mp, batchInd = fullBUTT_GPU(gen_obj, ref_map_base , target_midpoint, 
                                                            batch_size=100, cycles=200, input_z=None, 
                                                            rate=0.05, target_ep=[4,5,6,7], num_helices=4, 
                                                            oneRef=True, scale=100.0, z_size=12)

        out_ep, uInd = buttress_ep_from_z_mask_mp(gen_obj, output_z[-1], loss_mask[-1], loss_mp[-1], 
                                                  batchInd, max_mp_loss = 1e-3, max_loss_mask = 0.002, 
                                                  max_out=1000, print_stats= True, mask_first=True)

        fa, fa_reflect = align_generated_to_starting_ep(out_ep, current_quad_prez[uInd])

        #use the reflection that promotes the most movement along the guide points
        final_dr = determine_reflection(fa, fa_reflect, vg[uInd])

        #ensure that midpoint is close enough
        mpf = get_midpoint(final_dr,helices_desired=[2,3])
        mpfdb = np.linalg.norm(tmp[uInd].squeeze()-mpf,axis=1) < mp_deviation_limit

        final = final_dr[mpfdb]

        master_ep = np.concatenate((master_ep[uInd,:,:][mpfdb], final[:,4:,:]), axis=1)
        print(f'final pass filter' ,master_ep.shape[0])

        #remake ci here with new indices
        final_mp = get_midpoint(final, helices_desired=[2,3])
        dmp = np.linalg.norm(guide_points - np.expand_dims(final_mp,axis=1),axis=2)
        ci = np.argmin(np.abs(dmp),axis=1)

        end_dist = np.linalg.norm(guide_points[ci]-guide_points[-1],axis=1)
        outOfRoom = end_dist<next_mp_dist

        getOut = master_ep[outOfRoom]
        master_ep = master_ep[~outOfRoom]
        ci = ci[~outOfRoom]

        if len(getOut)>1:
            output_ep_list.append(getOut)

        if len(master_ep) < 2:
             roomToGrow = False
                
    return output_ep_list
        


#code to regenerate endpoints of dataset from fits
# dm_file = 'data/Fits_4H_dm_phi.npz'
# rr = np.load(dm_file, allow_pickle=True)
# X_train, y_train , featNames = [rr[f] for f in rr.files]
# = 'data/4H_dataset/models/'
# cmd.load(f'{model_direc}{y_train[0]}.pdb')
# cmd.save(f'output/test.pdb')

#endpoints for data set 
# Fits4H_file = 'data/Fits_4H.csv'
# dfRead = pd.read_csv(Fits4H_file)
# df1 = ft.prepData_Str(dfRead,rmsd_filter=100)
# df2 = ft.EndPoint(df1)
# ep = df2.to_numpy()[:,:24].astype(float).reshape((-1,8,3))
# X = ep
# np.savez_compressed('data/ep_for_X.npz', ep=X)



#prep endpoint dataset for use with easy use with Trilateration:
#Essentially identify 3 points on 1st two helices (Rotate/Translate to Z-plane) with 
#index mobile 1 at 0,0,0, target point in the positive z (trilateration assumtion)
#roughly 10% of z values of helices 3/4 are in the negative feild, with point enforced 4 positive

#distance map of ep dataset
#unsqueeze at two different dimensionsq to broadcast into matrix MX1 by 1XN to MXN 

def get_reference_input(batch=1000):
    rr = np.load(f'data/ep_for_X.npz', allow_pickle=True)
    X = [rr[f] for f in rr.files][0]

    dX = np.expand_dims(X,axis=1) - np.expand_dims(X,axis=2)
    dist = np.sqrt(np.sum(dX**2, 3))  #+ 1e-6) #this dataset is good 
    dist = dist.reshape((dist.shape[0],-1))
    mp_01 = get_midpoint(X,helices_desired=[0,1])
    mp_23 = get_midpoint(X,helices_desired=[2,3])
    #mp distance map
    ep_mp = np.hstack((X.reshape((-1,24)),mp_01,mp_23)).reshape(-1,10,3) #helix12mp=8  helix34mp=9


    #initiate array to hold endpoints
     #mp deviation from guide points

    #random sample starting endpoints to buttress
    refi_all = list(range(ep_mp.shape[0]))
    ref_ind = np.array(random.sample(refi_all , batch))
    #intialize set of points to match : 
    gep = generate_parabola(0, 40, h=[40], k=[40], num_points = 100)[0]

    start_hel = ep_mp[ref_ind ,:4,...]
    
    return start_hel, gep


#ahhh recode this without the global nonsense I suppose

# if tf.config.list_physical_devices('GPU'):
#     device_name = tf.test.gpu_device_name()
# else:
#     device_name = 'CPU'
rate=0.05
# if ~devtype.__eq__(device_name):
device_name = 'CPU'
print(f'device name {device_name}')

gen="data/BestGenerator"



with tf.device(device_name):
    gen_obj = ge.BatchRecon(gen)
    optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
output1=gen_obj.generate(z=12,batch_size=12) #example generator


# out_ep = build_protein_on_guide(start_hel, gep, batch=200, 
#                                  next_mp_dist = 11,mp_deviation_limit = 5)



# vp(out_ep[0],gep,max_out=100)

# sumL = 0

# for x in out_ep:
#     sumL += len(x)

# print(sumL)





