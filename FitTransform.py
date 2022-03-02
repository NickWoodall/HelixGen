import pandas as pd
import numpy as np

from math import cos,sin,tan,asin,acos,radians,sqrt,degrees,atan,copysign,atan2
import math
import util.RotationMethods as rm
import sys
import argparse

#--------------------------General Methods-----------------------------

def prepData_Str(data_frame, rmsd_filter = 101):
    """Convert to right data types csv file read from pandas. Filters rmsd, removes unused parameters."""
    df = data_frame.copy()
    df = df.loc[df['name'] != 'name']
    
    for col in df.columns:
        t="float"
        if "length" in col:
            t = "integer"
        elif 'name'  in col:
            continue
        df[col] = pd.to_numeric(df[col],downcast=t)
    
    #remove poor fitting rmsd
    df.insert(1, 'max_rmsd', df[df.filter(like='rmsd').columns].max(axis=1))
    df = df.loc[df['max_rmsd']<rmsd_filter]
    
    exParams = ['omega1','d', 'r1'] #d removes rmsd as well
    for x in exParams:
        df=df.drop(df.filter(like=x,axis=1).columns,axis=1)
        
    df = df.dropna()
    
    return df

def dist1(p1,p2):
    return np.linalg.norm(p2-p1)

def generate_distLabel(num_helices=4):
    
    numEndpoints = num_helices*2
    distLabels = []
    
    for x in range(1,numEndpoints+1):
        for y in range(x+1,numEndpoints+1):
            distLabels.append(f'd_{x}_{y}')
            
    return distLabels

def saveAsNumpy(df,name,direc=''):
    fullNumpy = df.to_numpy()
    names = fullNumpy[:,-1]
    feats = fullNumpy[:,:-1]
    fnames = df.columns[:-1]
    print(feats.shape)
    
    np.savez_compressed(f'{direc}{name}.npz',X_train=feats, y_train=names , featNames=fnames)

def saveAs_CSV(df,name,direc=''):
    
    df.to_csv(f'{direc}{name}')

def drop_extra_parameters(df,exParams = ['phi1']):
    dfOut = df.copy()
    for x in exParams:
        dfOut=dfOut.drop(dfOut.filter(like=x,axis=1).columns,axis=1)
        
    return dfOut


def df_to_dictList(dfIn,numHelices=4):
    
    labels = dfIn.columns
    npVal = dfIn.to_numpy()
    
    #remove name
    labels = labels[1:]
    npVal = npVal[:,1:]
    
    tot = int(len(labels)/4)
    #remove helix identifier (_X)
    keys = []
    for x in range(tot):
        keys.append(labels[x][:-2])
        
    npVal = npVal.reshape(npVal.shape[0],numHelices,-1)
    
    dListList = []
    
    for i in range(npVal.shape[0]):
        dList = []
        for j in range(npVal[i].shape[0]):
            dList.append(dict(zip(keys,npVal[i][j])))
        dListList.append(dList)
            
    return dListList


#------------------Conversion of HelixParameters to other Representations--------------------

def EndPoint(dfIn,num_helices=4):
    """Uses panda converts HelicalParameters to Endpoints."""
    
    df = dfIn.copy()
    


    zUnit = np.array([0,0,-1] ,dtype=np.float64)
    
    labels = ['x1','y1','z1','x2','y2','z2','x3','y3','z3',
                  'x4','y4','z4','x5','y5','z5','x6','y6','z6',
                  'x7','y7','z7','x8','y8','z8']
    
    
    dicList = []
    for i in range(df.shape[0]):
        
        pointList = []
        eulerList = []
        
        
        for j in range(num_helices):
            xNum = df.columns.get_loc(f'translate_x_{j+1}')
            yNum = df.columns.get_loc(f'translate_y_{j+1}')
            zNum = df.columns.get_loc(f'translate_z_{j+1}')
            
            pointList.append(np.array([df.iloc[i,xNum], df.iloc[i,yNum], df.iloc[i,zNum]],dtype=np.float64))

            phiNum = df.columns.get_loc(f'rotate_phi_{j+1}')
            thetaNum = df.columns.get_loc(f'rotate_theta_{j+1}')
            psiNum = df.columns.get_loc(f'rotate_psi_{j+1}')
            
            eulerList.append(np.array([df.iloc[i,phiNum], df.iloc[i,thetaNum], df.iloc[i,psiNum]],dtype=np.float64))
        
        endPointList = []
        for j in range(len(pointList)):
            
            R = rm.euler_to_R(eulerList[j][0],eulerList[j][1],eulerList[j][2])
            vec = rm.normalize(np.matmul(R,zUnit))
            lengthNum = df.columns.get_loc(f'length_{j+1}')
            length = df.iloc[i,lengthNum]
            
            risePerRes = 1.51
            multLen = int(length/2)
            revMultLen = multLen
            
            if length % 2 != 0:
                multLen += 1
                
            multLen = multLen*risePerRes
            revMultLen = revMultLen*risePerRes
                
            p1 = pointList[j]-revMultLen*vec
            p2 = pointList[j]+multLen*vec
            endPointList.extend(p1)
            endPointList.extend(p2)

        
        dicList.append(dict(zip(labels,endPointList)))
        
        for j in range(num_helices):
            dicList[-1][f'phi1_{j+1}'] = df.iloc[i,df.columns.get_loc(f'phi1_{j+1}')]
        dicList[-1]['name'] = df.iloc[i,df.columns.get_loc('name')]
        
        
    dfOut  = pd.DataFrame(dicList, columns=dicList[0].keys())
                
    return dfOut



def contactDist_EndPoint(dfIn,num_helices=4):
    """Converts Parameters form HelixParameter to endpoints then to endpoint distance map."""
    #Used for Representing Protein in GAN
    
#     labels = ['d_1_2', 'd_1_3', 'd_1_4', 'd_1_5', 'd_1_6', 'd_1_7', 'd_1_8', 'd_2_3', 'd_2_4', 
#                'd_2_5', 'd_2_6', 'd_2_7', 'd_2_8', 'd_3_4', 'd_3_5', 'd_3_6', 'd_3_7', 'd_3_8',
#                'd_4_5','d_4_6', 'd_4_7', 'd_4_8', 'd_5_6', 'd_5_7', 'd_5_8', 'd_6_7', 'd_6_8', 
#                'd_7_8']
    
    labels = generate_distLabel(num_helices=num_helices)
    
    iu1 = np.triu_indices(num_helices*2, 1) #indices for upper right triangle , no diagonal
    zUnit = np.array([0,0,-1] ,dtype=np.float64)
    
    df = dfIn.copy()
    dicList = []
    for i in range(df.shape[0]):
        
        pointList = []
        eulerList = []
        
        
        for j in range(num_helices):
            xNum = df.columns.get_loc(f'translate_x_{j+1}')
            yNum = df.columns.get_loc(f'translate_y_{j+1}')
            zNum = df.columns.get_loc(f'translate_z_{j+1}')
            
            pointList.append(np.array([df.iloc[i,xNum], df.iloc[i,yNum], df.iloc[i,zNum]],dtype=np.float64))

            phiNum = df.columns.get_loc(f'rotate_phi_{j+1}')
            thetaNum = df.columns.get_loc(f'rotate_theta_{j+1}')
            psiNum = df.columns.get_loc(f'rotate_psi_{j+1}')
            
            eulerList.append(np.array([df.iloc[i,phiNum], df.iloc[i,thetaNum], df.iloc[i,psiNum]],dtype=np.float64))
        
        endPointList = []
        for j in range(len(pointList)):
            
            R = rm.euler_to_R(eulerList[j][0],eulerList[j][1],eulerList[j][2])
            vec = rm.normalize(np.matmul(R,zUnit))
            lengthNum = df.columns.get_loc(f'length_{j+1}')
            length = df.iloc[i,lengthNum]
            
            risePerRes = 1.51
            revVec = -vec
            multLen = int(length/2)
            revMultLen = multLen
            
            if length % 2 != 0:
                multLen += 1
                
            multLen = multLen*risePerRes
            revMultLen = revMultLen*risePerRes
                
            p1 = pointList[j]-revMultLen*vec
            p2 = pointList[j]+multLen*vec
            endPointList.append(p1)
            endPointList.append(p2)
                
        distMap = np.zeros((len(endPointList),len(endPointList)),dtype=np.float64)
        for x in range(len(endPointList)):
            for y in range(len(endPointList)):
                distMap[x][y] = dist1(endPointList[x],endPointList[y])       
        

        distC = distMap[iu1]

        
        dicList.append(dict(zip(labels,distC)))
        
        for j in range(num_helices):
            dicList[-1][f'phi1_{j+1}'] = df.iloc[i,df.columns.get_loc(f'phi1_{j+1}')]
        dicList[-1]['name'] = df.iloc[i,df.columns.get_loc('name')]
        
        
    dfOut  = pd.DataFrame(dicList, columns=dicList[0].keys())
                
    return dfOut


# In[3]:


def contact_Dist_Dihedral(dfIn,num_helices=4):
    """Converts HelixParameters to contact_dist(helix midpoint) and dihedrals angles between each helix vector."""
    #Used for clustering
    
    labels = ['d_1_2', 'd_1_3', 'd_1_4', 'd_2_3', 'd_2_4', 'd_3_4',
               'a_1_2', 'a_1_3', 'a_1_4', 'a_2_3', 'a_2_4', 'a_3_4']
    
    iu1 = np.triu_indices(num_helices, 1) #indices for upper right triangle , no diagonal
    
    df = dfIn.copy()
    dicList = []
    for i in range(df.shape[0]):
        
        pointList = []
        eulerList = []
        
        
        for j in range(num_helices):
            xNum = df.columns.get_loc(f'translate_x_{j+1}')
            yNum = df.columns.get_loc(f'translate_y_{j+1}')
            zNum = df.columns.get_loc(f'translate_z_{j+1}')
            
            pointList.append(np.array([df.iloc[i,xNum], df.iloc[i,yNum], df.iloc[i,zNum]],dtype=np.float64))

            phiNum = df.columns.get_loc(f'rotate_phi_{j+1}')
            thetaNum = df.columns.get_loc(f'rotate_theta_{j+1}')
            psiNum = df.columns.get_loc(f'rotate_psi_{j+1}')
            
            eulerList.append(np.array([df.iloc[i,phiNum], df.iloc[i,thetaNum], df.iloc[i,psiNum]],dtype=np.float64))
        
        distMap = np.zeros((len(pointList),len(pointList)),dtype=np.float64)
        diheMap = np.zeros((len(eulerList),len(eulerList)),dtype=np.float64)
        for x in range(len(pointList)):
            for y in range(len(pointList)):
                distMap[x][y] = dist1(pointList[x],pointList[y])
                diheMap[x][y] = rm.dihedral_from_euler(pointList[x],pointList[y],eulerList[x],eulerList[y])     
        
        dist = distMap[iu1]
        dihe = diheMap[iu1]
        
        dist_dihe=np.hstack((dist,dihe))
        
        dicList.append(dict(zip(labels,dist_dihe)))
        
        for j in range(num_helices):
            dicList[-1][f'phi1_{j+1}'] = df.iloc[i,df.columns.get_loc(f'phi1_{j+1}')]
            dicList[-1][f'length_{j+1}'] = df.iloc[i,df.columns.get_loc(f'length_{j+1}')]
        dicList[-1]['name'] = df.iloc[i,df.columns.get_loc('name')]
        
        
    dfOut  = pd.DataFrame(dicList, columns=dicList[0].keys())
    #d
                
    return dfOut




def convert_to_endpoints_distmap(inFile,outFile,rmsd_max = 10):
    """Converst csv save of HelixParameters to endpoint distance map for GAN"""
    
    dfRead = pd.read_csv(inFile)
    df1 = prepData_Str(dfRead,rmsd_filter=rmsd_max)
    df2 = contactDist_EndPoint(df1)
    saveAsNumpy(df2,outFile)
    
def convert_to_endpoints(inFile,outFile,rmsd_max = 10):
    """Converts csv save of HelixParameters to endpoint distance map for GAN"""
    
    dfRead = pd.read_csv(inFile)
    df1 = prepData_Str(dfRead,rmsd_filter=rmsd_max)
    df2 = EndPoint(df1)
    saveAsNumpy(df2,outFile)
    
def convert_to_distdihe(inFile,outFile,rmsd_max = 10):
    """Converts csv save of HelixParameters to endpoint distance map for GAN"""
    
    dfRead = pd.read_csv(inFile)
    df1 = prepData_Str(dfRead,rmsd_filter=rmsd_max)
    df2 = contact_Dist_Dihedral(df1)
    saveAsNumpy(df2,outFile)
    


            
            
            
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Converts Straight Helix Parameters into other formats. -d for GAN training")
    

    parser.add_argument("infile", help="CSV of Helix Parameters")
    parser.add_argument("outfile", help="Output File Name")

    parser.add_argument("-e", "--endpoints", help="Output Data as Endpoints / Phi", action="store_true")
    parser.add_argument("-d", "--distancemap", help="Output Data as Distance Map / Phi", action="store_true")
    parser.add_argument("-c", "--distdihe_cluster", help="Output Data as Distance/Dihedrals", action="store_true")
    parser.add_argument("-r", "--rmsd_cut", help="Cut off for RMSD of Fitting", default=10.0, type=float)
    args = parser.parse_args()


    if args.endpoints:
        convert_to_endpoints(args.infile,args.outfile, rmsd_max=args.rmsd_cut)
        if args.distancemap:
            convert_to_endpoints_distmap(args.infile,args.outfile, rmsd_max=args.rmsd_cut)
            
    
    elif args.distdihe_cluster:
         convert_to_distdihe(args.infile,args.outfile, rmsd_max=args.rmsd_cut)
    else:
        convert_to_endpoints_distmap(args.infile,args.outfile, rmsd_max=args.rmsd_cut)
        

        

    




