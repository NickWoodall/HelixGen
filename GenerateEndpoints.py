import sys
import os
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

import tensorflow as tf

import joblib
from sklearn.manifold import MDS
import argparse

import HelixFit as hf
import FitTransform as ft

zero_ih = nu.npose_from_file('util/zero_ih.pdb')
tt = zero_ih.reshape(int(len(zero_ih)/5),5,4)
stub = tt[7:10].reshape(15,4)




###-------Generate Endpoints

class BatchRecon():
    
    def __init__(self, name,size=8, h4_feats=28):
        """Load generator to make endpoints. Make sure MinMaxScaler is in the same directory with _mm added to end."""
        self.name = name
        self.g=tf.keras.models.load_model(f'{name}.h5')
        self.mm = joblib.load(f'{name}_mm.gz')
        
        self.zero_ih = zero_ih
        
        self.size = size
        self.feats = h4_feats
        self.iu1 = np.triu_indices(self.size, 1)

    
    def generate(self,z,batch_size=24):
        """Generate distance map."""
        
        self.batch_size = batch_size
        z_size = z

        input_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)
        
        g_output = self.g(input_z, training=False)
        g_output = tf.reshape(g_output, (batch_size, self.feats))
    
        g_output = g_output.numpy()
        self.g_output = self.mm.inverse_transform(g_output)

        return self.g_output
            
    def MDS_reconstruct_(self):
        """Use distance map to recover endpoints via sklearn MDS"""
        
        self.reconsMDS_ = []
        self.endpointDict_list_ = []
        
        
        labels = ['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4',
                  'x5','y5','z5','x6','y6','z6','x7','y7','z7','x8','y8','z8']

        for x in self.g_output:
            distMap = np.zeros((self.size,self.size),dtype=np.float64)
            embedding = MDS(n_components=3,dissimilarity='precomputed')
            distMap[self.iu1] = x
            a = distMap + distMap.T
            pVec = embedding.fit_transform(a)
            self.reconsMDS_.append(pVec)
        
        return self.reconsMDS_
    
    def reconstructionError(self):
        """Error of reconstruction between endpoints and distance map."""
        
        self.errorList = []
        
        def dist1(p1,p2):
            return np.linalg.norm(p2-p1)
        
        def point_to_dist(pVec):
            contactMap = np.zeros((self.size,self.size),dtype=np.float64)
            for index in range(len(pVec)):
                for j in range(len(pVec)):
                    contactMap[index][j] = dist1(pVec[index],pVec[j])
            return contactMap
        
        
        for x in range(len(self.reconsMDS_)):
            reconDistMap = point_to_dist(self.reconsMDS_[x])[self.iu1]
            errorMSE = np.mean(np.square(reconDistMap-self.g_output[x]))
            self.errorList.append(errorMSE)
            
        return np.mean(self.errorList)
    
    def to_HelicalProtein(self):
        """Convert to helical protein object. Mostly to convert to parameters."""
        
        self.hpList = []
        
        for x in self.reconsMDS_:
            self.hpList.append(hf.HelicalProtein.from_endpoints(x))
        
        self.hpLabel = self.hpList[0].getLabel_().split(",")
        
        self.expFit = []
        
        for x in range(len(self.hpList)):
            self.expFit.append(self.hpList[x].export_fits_().split(","))
            
        return self.expFit, self.hpLabel
            
            
    def to_npose(self):
        """Convert endpoints to straight ideal helices. Npose of the form atom position list [x,y,z,1]"""

        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0: 
                return v
            return v / norm

        def angle_two_vectors(v1,v2):
            #assuming normalize
            #https://onlinemschool.com/math/library/vector/angl/

            # cos α = 	a·b
            #         |a|·|b|

            dp = np.dot(v1,v2)
            return  acos(dp)
        
        def ep_to_xform(p1,p2):
            """Endpoints to rotation matrix [xform] to rotate standard helix into position. Returns rotated helix"""
            zUnit  = np.array([0,0,-1])
            vector = normalize(p2 - p1)

            axisRot = normalize(np.cross(vector,zUnit))
            ang = angle_two_vectors(vector, zUnit)

            #mod here to one decimal?
            length = int(np.round(np.linalg.norm(p2-p1)/1.51,decimals=2))
            halfLen = int(length/2)
            aRot=np.hstack((axisRot,[1]))

            mp = p1+vector*float(halfLen/length)*np.linalg.norm(p2-p1)


            #global variable
            len_zero_ih = int(len(self.zero_ih)/5)

            hLen = int((len_zero_ih-length)/2)
            xform1 = nu.xform_from_axis_angle_rad(aRot,-ang)
            xform1[0][3] = mp[0] 
            xform1[1][3] = mp[1] 
            xform1[2][3] = mp[2]
            zI = np.copy(self.zero_ih)

            #aligned_pose = nu.xform_npose(xform1, zI)

            if length % 2 == 1:
                aligned_pose = nu.xform_npose(xform1, zI[(hLen*5):(-hLen*5)] )
            else:
                aligned_pose = nu.xform_npose(xform1, zI)
                aligned_pose = aligned_pose[((hLen)*5):(-(hLen+1)*5)]

            return aligned_pose, length
    
        
        self.npose_list = []
        self.helixLength_list = [] 
        #maintain helix length list to keep track of helices in npose_list
        
        
        for y in range(len(self.reconsMDS_)):
            apList = np.array(np.empty((0,4), np.float32))
            #hardcoded 4
            self.helixLength_list.append([])
            for x in range(0,len(self.reconsMDS_[y]),2):
                t,h_length = ep_to_xform(self.reconsMDS_[y][x],self.reconsMDS_[y][x+1])
                self.helixLength_list[y].append(h_length)
                apList = np.vstack((apList,t))
                
            self.npose_list.append(apList)

        return self.npose_list
    
    
    
class EP_Recon():
    """Parrallel Class to BatchRecon to loading endpoints from elsewhere and run LoopedEndpoints"""
    
    def __init__(self, fname, direc=None):
        
        rr = np.load(f'{fname}.npz', allow_pickle=True)
        self.endpoints_list = [rr[f] for f in rr.files][0]
    
    def to_npose(self):

        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0: 
                return v
            return v / norm

        def angle_two_vectors(v1,v2):
            #assuming normalize
            #https://onlinemschool.com/math/library/vector/angl/

            # cos α = 	a·b
            #         |a|·|b|

            dp = np.dot(v1,v2)
            return  acos(dp)
        
        def ep_to_xform(p1,p2):
            zUnit  = np.array([0,0,-1])
            vector = normalize(p2 - p1)

            axisRot = normalize(np.cross(vector,zUnit))
            ang = angle_two_vectors(vector, zUnit)


            length = int(np.round(np.linalg.norm(p2-p1)/1.51,decimals=0))
            halfLen = int(length/2)
            aRot=np.hstack((axisRot,[1]))

            mp = p1+vector*float(halfLen/length)*np.linalg.norm(p2-p1)


            #global variable
            len_zero_ih = int(len(zero_ih)/5)

            hLen = int((len_zero_ih-length)/2)
            xform1 = nu.xform_from_axis_angle_rad(aRot,-ang)
            xform1[0][3] = mp[0] 
            xform1[1][3] = mp[1] 
            xform1[2][3] = mp[2]
            zI = np.copy(zero_ih)

            #aligned_pose = nu.xform_npose(xform1, zI)

            if length % 2 == 1:
                aligned_pose = nu.xform_npose(xform1, zI[(hLen*5):(-hLen*5)] )
            else:
                aligned_pose = nu.xform_npose(xform1, zI)
                aligned_pose = aligned_pose[((hLen)*5):(-(hLen+1)*5)]

            return aligned_pose, length
    
        
        self.npose_list = []
        self.helixLength_list = [] 
        
        
        for y in range(len(self.endpoints_list)):
            apList = np.array(np.empty((0,4), np.float32))
            #hardcoded 4
            self.helixLength_list.append([])
            for x in range(0,len(self.endpoints_list[y]),2):
                t,h_length = ep_to_xform(self.endpoints_list[y][x],self.endpoints_list[y][x+1])
                self.helixLength_list[y].append(h_length)
                apList = np.vstack((apList,t))
                
            self.npose_list.append(apList)

        return self.npose_list


def generate_endpoints(genName='data/BestGenerator',outName='gen_ep',batch_size=32,z=12):
    br = BatchRecon(name=genName)
    br.generate(z,batch_size=batch_size)
    endpoint_list = br.MDS_reconstruct_()
    ft.saveAsNumpy(endpoint_list,outName)

def generate_straight_helices(genName='data/BestGenerator',outName='gen_ep.pdb',batch_size=32,z=12):
    br = BatchRecon(name=genName)
    br.generate(z,batch_size=batch_size)
    br.MDS_reconstruct_()
    br.to_npose()
    nu.dump_npdb(br.npose_list,outName)
    
def generate_dist_dihe(genName='data/BestGenerator',outName='gen_ep',batch_size=32,z=12):
    br = BatchRecon(name=genName)
    br.generate(z,batch_size=batch_size)
    br.MDS_reconstruct_()
    fits, label = br.to_HelicalProtein()
    df = pd.DataFrame(data=fits, columns=label)
    df = ft.prepData_Str(df)
    df1 = ft.contact_Dist_Dihedral(df)
    ft.saveAsNumpy(df1,outName)






if __name__ == "__main__":
    #
    
    parser = argparse.ArgumentParser(description="Generates endpoints by Default. Also Straight Helices and Dist_Dihe for Clustering")
    
    parser.add_argument("-b","--batch", help="Number of Endpoints to Generate",  default=32, type=int)
    parser.add_argument("-i", "--infile", help="Location of generator network, No File extension. MinMax name needs to be of form {name}_mm", default="data/FullSet")
    parser.add_argument("-o", "--outfile", help="output file name", default="generated_ep")
    parser.add_argument("-s", "--straight_helix", help="Output as straight helices", action="store_true")
    parser.add_argument("-d", "--dist_dihe", help="Output as distance_dihedral", action="store_true")
    parser.add_argument("-z", "--genInputSize", help="Size of vector to input to generator",default=12, type=int)
    args = parser.parse_args()
    

    if args.straight_helix:
        generate_straight_helices(genName=args.infile,outName=f'{args.outfile}.pdb',z=args.genInputSize)
    elif args.dist_dihe:
        generate_dist_dihe(genName=args.infile,outName=args.outfile,batch_size=args.batch,z=args.genInputSize)
    else:
        generate_endpoints(genName=args.infile,outName=args.outfile,batch_size=args.batch,z=args.genInputSize)
                        
    
#generate_dist_dihe('data/BestGenerator','Clustering/data/test',batch_size=5024)
    
