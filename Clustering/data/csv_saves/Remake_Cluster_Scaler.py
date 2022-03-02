
import numpy as np
import pandas as pd
import pickle
import os

import joblib
import sklearn.preprocessing as skp
import argparse

def vec2string(vecIn):
    vecString = f''
    for x in vecIn:
        vecString = f'{vecString}{x:0.6f},'
        
    return vecString[:-1]


def cluster_csv_save(clusterLoad = '../data/refData', outName='refClus'):
    
    with np.load(f'{clusterLoad}.npz', allow_pickle=True) as rr:
            y_, y_train, X_train, featNames = [rr[f] for f in rr.files]
            
    strOut = []
    for name,clus,feats in zip(y_train,y_,X_train):
        featString = vec2string(feats)
        strOut.append(f'{name},{clus},{featString}')
        
    featStr = ""
    for x in featNames:
        featStr = f'{featStr}{x},'
    featStr = featStr[:-1]

    with open(f'{outName}.csv', "w") as f:
        f.write(f'name,cluster,{featStr}\n')
        for x in strOut:
            f.write(f'{x}\n')

def remake_clusters(outName, csv_name='refClus.csv'):
    
    with open(csv_name,"r") as f:
        data = f.readlines()
        
    featNames = []
    featNames = data[0].strip().split(",")[2:]
    y_new = []
    y_train_new = []
    X_train_new = []

    for x in range(1,len(data)):
        dataLine = data[x].strip().split(",")
        y_train_new.append(dataLine[0])
        y_new.append(dataLine[1])
        X_train_new.append(dataLine[2:])
        
    yTn = np.array(y_train_new)
    yn = np.array(y_new,dtype=np.int32)
    Xn = np.array(X_train_new,dtype=np.float32)
    
    np.savez_compressed(f'{outName}.npz', y_ = yn , y_train = yTn, 
                            X_train=Xn, featNames = featNames)

def scaler_csv_save(outName,name='../refData_scaler.gz'):
    saveScaler = joblib.load(name)
    
    mean_=vec2string(saveScaler.mean_)
    scale_=vec2string(saveScaler.scale_)
    n_samples_seen_ = str(saveScaler.n_samples_seen_)
    var_ = vec2string(saveScaler.var_)
    n_features_in_=str(saveScaler.n_features_in_)
    
    with open(f'{outName}.csv', "w") as f:
        f.write(f'{n_samples_seen_}\n')
        f.write(f'{mean_}\n')
        f.write(f'{scale_}\n')
        f.write(f'{var_}\n')
        f.write(f'{n_features_in_}\n')
        
def remake_scaler_csv(scalerIn, outName):
    
    with open(scalerIn,'r') as f:
        data = f.readlines()
    
    n_samples_seen_ = int(data[0])
    mean_ = np.array(data[1].strip().split(","),dtype=np.float32)
    scale_ = np.array(data[2].strip().split(","),dtype=np.float32)
    var_ = np.array(data[3].strip().split(","),dtype=np.float32)
    n_features_in_ = int(data[4].strip())
    
    scale = skp.StandardScaler()
    scale.n_samples_seen_ = n_samples_seen_
    scale.mean_ = mean_
    scale.scale_ = scale_
    scale.var_ = var_
    scale.n_features_in_ = n_features_in_
    
    joblib.dump(scale, f'{outName}_scaler.gz')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remake original cluster objects/scaler from csv files.")
    
    parser.add_argument("-n", "--clus_name", default="refData")
    parser.add_argument("-s", "--save_as_csv", help="save reference data as csv", action="store_true")
    
    
    
    args = parser.parse_args()                    
                        
    if args.save_as_csv:
        scaler_csv_save('refClus_scaler', name='../refData_scaler.gz')
        cluster_csv_save(clusterLoad = '../refData', outName='refClus')
    else:
        remake_scaler_csv(f'refClus_scaler.csv',args.clus_name)
        remake_clusters(args.clus_name, csv_name='refClus.csv')
        


