
from pymol import cmd, stored, selector
import random
import os
import numpy as np
from sklearn.cluster import KMeans
import time
import re
import argparse

import pandas as pd

class AlignCluster():
    
    def __init__(self, clusterData='data/refData', loadLimit=100,dModel='../data/bCov_4H_dataset/BCov_Models',
                 curClus=0, saveDirec = 'data/clusterRef'):
        """Aligns based on reflections. Since proteins are chiral and Distance Dihedrals angles lose this chirality."""
        
        
        
        self.dModel = dModel
        self.royalCouple = None # class representatives (1+reflection)
        self.batchEnd= 0 # keeps track of where in cluster file list
        self.batchStart = 0
        self.cluster = [] #cluster file list. For one cluster at time
        self.curClus = curClus
        
        self.loadLimit = loadLimit
        self.outlierCutoff = 4.5 # rmsd filter for proteins that align poorly to both class reps
        
        with np.load(f'{clusterData}.npz', allow_pickle=True) as rr:
            self.y_, self.y_train, self.X_train, self.featNames = [rr[f] for f in rr.files]
        
        self.cluster_labels = np.unique(self.y_)
        self.n_clusters = self.cluster_labels.shape[0]
        
        self.fileList = os.listdir(self.dModel)
        self.saveDirec = saveDirec
        
        
    def align_cluster(self,clusterNum=1, clusterRepTry=30, limit=150):
        """Pulls files, finds representative, aligns other proteins to representatives. Saves SessionFile"""
        
        self.getClusterFiles(clusterNum,limit=limit)
        print(f'Finding Cluster Representative for Cluster {clusterNum}.')
        self.findClusterRep(limit=clusterRepTry)
        print(f'Aligning to Cluster Representative.')
        self.align_to_royal_couple(saveSession=True)


    def getClusterFiles(self, cluster, limit=100000):

        self.curClus = cluster
        
        index = np.where(self.y_==self.curClus)
        self.cluster = self.y_train[index]
        
        if limit < len(self.cluster):
            self.cluster = random.choices(self.cluster,k=limit)
   
    
    def updateCMD(self,cluster=None, clear=True, keepRoyalCouple=True):
        """Keep CMD, pymol session, clear of too many proteins. Keeps Class Rep."""
        
        if clear:
            if self.royalCouple is not None:
                if keepRoyalCouple:
                    cmd.delete(f'not {self.royalCouple[0]} or not {self.royalCouple[1]}')
                else:
                    cmd.delete("all")
            else:
                cmd.delete("all")
                
        if cluster is not None:
            for x in cluster:
                if self.royalCouple is not None:
                    if x == self.royalCouple[0] or x == self.royalCouple[1]:
                        continue
                cmd.load(f'{self.dModel}/{x}.pdb')
                
    def resetBatch(self):
        self.batchEnd= 0 # keeps track of where in cluster file list
        self.batchStart = 0
        self.updateCMD()
        
    def saveClassReference(self):
        
        for y in self.royalCouple:
            cmd.save(f'{self.saveDirec}/cluster{self.curClus}__{y}.pdb',selection=y)
        
    def loadClassReference(self,curCluster=None):
        #requires only two present or will fail
        
        if curCluster is not None:
            self.curClus = curCluster
        
        fileList = os.listdir(self.saveDirec)
        
        self.royalCouple =[]
        
        
        for x in fileList:
            if f'cluster{self.curClus}__' in x:
                removeExt = x[:-4]
                self.royalCouple.append(removeExt)
            if len(self.royalCouple) >= 2:
                break
                
        for x in self.royalCouple:
            cmd.load(f'{self.saveDirec}/{removeExt}.pdb')
        
                
    def saveSession(self):
        
        self.classSelection() 
        cmd.save(f'{self.saveDirec}/aligned_{self.curClus}.pse')
        
    
    def batchLoad(self):
        
        self.updateCMD(clear=True, keepRoyalCouple=True)
        
        self.batchStart = self.batchEnd
        
        if self.loadLimit+self.batchStart > len(self.cluster):
            self.batchEnd = len(self.cluster)
        else:
            self.batchEnd = self.loadLimit+self.batchStart
        
        for x in range(self.batchStart,self.batchEnd):
            if self.royalCouple is not None:
                if self.cluster[x]  == self.royalCouple[0] or self.cluster[x]  == self.royalCouple[1]:
                    continue
                cmd.load(f'{self.dModel}/{self.cluster[x]}.pdb')
            else:
                cmd.load(f'{self.dModel}/{self.cluster[x]}.pdb')
            
        
        return self.batchEnd


    def findClusterRep(self, n_clusters=2, limit=30):
        """All by all alignment of helical proteins. KMeans fit all by all rmsd for Class Rep."""

        if len(self.cluster) > limit:
            cluster = random.sample(self.cluster,k=limit)
        else:
            cluster = self.cluster
            
        self.updateCMD(cluster=cluster)

        rmsList = np.zeros((len(cluster),len(cluster)))
        
        #all by all alignment using pair fit based on helical residues from center of each helix
        for i,val in enumerate(cluster):
            refProt = val
            for j, prot2 in enumerate(cluster):
                rmsList[i][j]= AlignCluster.pairFit_Helix(refProt,prot2)
        
        #cluster by kmeans
        km = KMeans(n_clusters = n_clusters, init='k-means++', n_init=10,  max_iter=4000, tol=1e-04)
        y_ = km.fit_predict(rmsList)

        cluster_labels = np.unique(y_)
        centerList = np.zeros((n_clusters,),dtype=np.int32)

        
        for j in cluster_labels:
            #should return list of distances to center of j'th cluster, argsort gives closest
            #used as a reference for pair_fit
            d = km.transform(rmsList)[::][:,j]
            centerList[j] = int(np.argsort(d)[0])

#         rmsOut = np.ones((len(cluster),))

#         for i, val in enumerate(cluster):
#             rmsOut[i]=pairFit_Helix(cluster[centerList[y_[i]]],val)
        self.royalCouple = []
        
        for x in centerList:
            self.royalCouple.append(cluster[x])
    
        return self.royalCouple
    
    def align_to_royal_couple(self,saveSession=False):
        #hardcoded for 2, align to cluster representative or reflected cluster representative
        self.rmsOut = []
        self.reflectionList = np.zeros((len(self.cluster),))
        
        self.updateCMD()
        self.batchEnd = self.batchLoad()
        
        for i,x in enumerate(self.cluster):
            
            if i >=self.batchEnd:
                if saveSession:
                    self.classSelection() 
                    cmd.save(f'{self.saveDirec}aligned_{self.curClus}.pse')
                        
                #delete cmd,reload royal couple, load next loadLimit, update index
                self.batchEnd = self.batchLoad() 

            rms = []
            refPoint = False
            
            refCount = 0
            for y in self.royalCouple:
                if y==x:
                    refPoint = True
                    break
                refCount += 1
                rms.append(AlignCluster.pairFit_Helix(y,x))
                
                
            if refPoint:
                self.reflectionList[i] = refCount
                self.rmsOut.append(0.0)
                print(refPoint,x)
                continue
                
            if rms[0]<rms[1]:
                self.rmsOut.append(AlignCluster.pairFit_Helix(self.royalCouple[0],x))
                if rms[0]<self.outlierCutoff:
                    self.reflectionList[i] = 0
                else:
                    self.reflectionList[i]=2
                    
                self.rmsOut.append(rms[0])
            elif rms[1]<rms[0]:
                self.rmsOut.append(AlignCluster.pairFit_Helix(self.royalCouple[1],x))
                if rms[1]<self.outlierCutoff:
                    self.reflectionList[i] = 1
                else:
                    self.reflectionList[i]=2
                    
                self.rmsOut.append(rms[1])
            else:
                #denotes outlier
                self.reflectionList[i] = 2
                self.rmsOut.append(min(rms))
                
                
        if saveSession:
            self.classSelection() 
            cmd.save(f'{self.saveDirec}/aligned_{self.curClus}.pse')
                
        return self.reflectionList


    
    def classSelection(self):
    
        selClass1 = ""
        selClass2 = ""
        selOutlier = ""
        
        for x in range(self.batchStart, self.batchEnd):
            if self.reflectionList[x] == 0:
                selClass1 = f'{selClass1}, {self.cluster[x]}'
            elif self.reflectionList[x] == 1:
                selClass2 = f'{selClass2}, {self.cluster[x]}'
            else:
                selOutlier = f'{selOutlier}, {self.cluster[x]}'
                
        selClass1 = selClass1[1:]
        selClass2 = selClass2[1:]
        selOutlier = selOutlier[1:]
        
        
        if selClass1:
            cmd.select('class1', selClass1)
        if selClass2:
            cmd.select('class2', selClass2)
        if selOutlier:
            cmd.select('outlier',selOutlier)

    @staticmethod
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
            print(f'Check{name}: Helices not as expected. Rerun unlikely to re-occur or load error')

        return helixRes
        
    @staticmethod
    def list_alignHelices(name1, name2, helixNum=1):

        hList1 = AlignCluster.get_HelixList(name1)
        hList2 = AlignCluster.get_HelixList(name2)
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
    
    @staticmethod
    def hSel(hListList, name):

        resString = ""

        for x in hListList:
            resString = f'{resString}+{x[0]}-{x[-1]}'

        resString = resString[1:]

        return f'{name} and resi {resString} and name CA'
    
    
    
    @staticmethod
    def pairFit_Helix(prot1,prot2):
        
        #moves prot2 onto prot1


        rmsList = []
        errorList = []

        p1_h1, p2_h1= AlignCluster.list_alignHelices(prot1,prot2, helixNum=1)
        p1_h2, p2_h2= AlignCluster.list_alignHelices(prot1,prot2, helixNum=2)
        p1_h3, p2_h3= AlignCluster.list_alignHelices(prot1,prot2, helixNum=3)
        p1_h4, p2_h4= AlignCluster.list_alignHelices(prot1,prot2, helixNum=4)

        p1List = [p1_h1,p1_h2,p1_h3,p1_h4]
        p2List = [p2_h1,p2_h2,p2_h3,p2_h4]
        
        rms = cmd.pair_fit(AlignCluster.hSel(p2List, prot2),AlignCluster.hSel(p1List, prot1))
        return rms

    
class PymolArt():
    def __init__(self, num_clusters = 26,direc='data/clusterRef',loadLimit=200):
        """Arranges Proteins in Pymol Session. Currently for view call representatives of clustering"""
        
        self.direc = direc
        
        self.loadLimit = loadLimit
        self.fileList = []
         
    def getFiles(self):
        
        fileList = os.listdir(self.direc)
        
        clustersLoaded = []
        
        count = 0
        
        for x in fileList:
            if count > self.loadLimit:
                break
            if not x.endswith('.pdb'):
                continue
            count += 1

            removeExt = x[:-4] #remove .pdb
            self.fileList.append(removeExt)
            
        return self.fileList
            
        
        
    
    def getClusterRepFiles(self, limit=200,skip=True):

        fileList = os.listdir(self.direc)
        
        clustersLoaded = []
        
        count = 0
        for x in fileList:
            if count > limit:
                break
            
            if not x.endswith('.pdb'):
                continue
                
            count += 1
            
            cNum = x.split('__')[0]
            temp1 = int(re.findall(r'\d+', cNum)[0])
            
            if temp1 in clustersLoaded:
                continue
            
            clustersLoaded.append(temp1)
            
            removeExt = x[:-4] #remove .pdb
            self.fileList.append(removeExt)
        
    
    def updateCMD(self,clear=True):
        
        if clear:
            cmd.delete("all")
        for x in self.fileList:
            cmd.load(f'{self.direc}/{x}.pdb')
                
    def all_by_all_rms(self):
        """All by all alignment using pair fit (see AlignCluster), get RMS to sort"""
                
        rmsList = np.zeros((len(self.fileList),len(self.fileList)))

        #all by all alignment using pair fit based on helical residues from center of each helix
        for i,val in enumerate(self.fileList):
            refProt = val
            for j, prot2 in enumerate(self.fileList):
                rmsList[i][j]= AlignCluster.pairFit_Helix(refProt,prot2)
                
        self.rmsList = rmsList

        return self.rmsList
    
    def sortRMS(self):
        
        df = pd.DataFrame(self.rmsList)
        df=df.sort_values(by=list(range(len(self.fileList))))
        
        indexList = df.index.values
        
        self.fileList[:] = [self.fileList[i] for i in indexList]
        
    
    def makeArray(self,rows=1, spacer=30):
        
        
        refProt = self.fileList[0]
        
        #align to one protein
        for x in range(1,len(self.fileList)):
            AlignCluster.pairFit_Helix(refProt,self.fileList[x])
            
        columns = int(np.ceil(len(self.fileList)/rows))
        
        
        for i,val in enumerate(self.fileList):
            cmd.translate([(i%columns)*spacer,1.25*spacer*(int(i/columns)),0],val,camera=0)
    
    def saveSession(self,fName='test'):
        
        cmd.save(f'{self.direc}/{fName}.pse')
        
        

def align_specific_cluster(dModel, clusterRepTry=10, limit=30, clusterData='data/refData', 
                           saveDirec='data/clusterRef/',clusterNum=0):
    ac = AlignCluster(clusterData=clusterData,dModel=dModel, saveDirec = saveDirec)
    ac.align_cluster(clusterNum=clusterNum,clusterRepTry=clusterRepTry,limit=limit)
    ac.saveClassReference()

def align_all_clusters(dModel, clusterRepTry=10, limit=30,clusterData='data/refData', 
                        saveDirec='data/clusterRef/'):
    ac = AlignCluster(clusterData=clusterData,dModel=dModel, saveDirec = saveDirec)
    nclus = ac.n_clusters
    for x in range(nclus):
        ac = AlignCluster(clusterData=clusterData,dModel=dModel, saveDirec = saveDirec)
        ac.align_cluster(clusterNum=x,clusterRepTry=clusterRepTry,limit=limit)
        ac.saveClassReference()
        
def array_clusters_pymol(outName= 'arrayed_output',saveDirec='data/clusterRef', numClusters=26, loadLimit = 100, rows=7):
    PA = PymolArt(direc=saveDirec, Limit=loadLimit,num_clusters = numClusters)
    PA.getClusterRepFiles()
    PA.updateCMD()
    PA.saveSession(fName=outName)

    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description ="Visualize clusters.")
    
    parser.add_argument("-i","--inDirec", help="Directory to save or load Cluster Representatives.", default='data/clusterRef')
    parser.add_argument("-m","--modelDirec", help="Location of pdbs files that have been clustered.")
    parser.add_argument("-o", "--outFile", help="Output File Name.", default="array_output")

    parser.add_argument("-c", "--cluster_data", help="Cluster file numpy array", default="data/refData")
    parser.add_argument("--clusterRepTry",help="Number of trials to find cluster representative",default=10)
    parser.add_argument("--numAlign", help="Number of examples to align to cluster representative",default=30)
    parser.add_argument("-s", "--singleAlign", help="Cluster number to align a single cluster. Initiates aligning only this cluster. Requires -m,-n", action="store_true")
    parser.add_argument("-n", "--cluster_num", help="Cluster number to align a single cluster. Initiates aligning only this cluster. Requires -m", type=int)
    parser.add_argument("-a","--all_cluster", help="align all clusters. Requires -m", action="store_true")
    parser.add_argument("-p","--pymol_array", help="array cluster representatives in pymol session. Requires -u", action="store_true")
    parser.add_argument("-r","--rows", help="rows to array proteins in", default=7, type=int)
    parser.add_argument("-u","--array_cluster", help="number of clusters to array in pymol session")
    parser.add_argument("--array_generic", help="load pdbs from direc instead of cluster representatives", action="store_true")
    
    args = parser.parse_args()
    
    if args.all_cluster:
        align_all_clusters(args.modelDirec, clusterRepTry=args.clusterRepTry, limit=args.numAlign,
                           clusterData=args.cluster_data, saveDirec=args.inDirec)
        
    elif args.singleAlign:
        align_specific_cluster(args.modelDirec, clusterRepTry=args.clusterRepTry, limit=args.numAlign,
                               clusterData=args.cluster_data, saveDirec=args.inDirec,clusterNum=args.cluster_num)
    elif args.pymol_array:
        
        PA = PymolArt(direc=args.inDirec,loadLimit=100,num_clusters = args.array_cluster)
        if args.array_generic:
            PA.getFiles()
        else:
            PA.getClusterRepFiles()
        PA.updateCMD()

        PA.makeArray(rows=args.rows)
        PA.saveSession(fName=args.outFile)
        
    
#bCov_Model_direc = '../../../HelicalGenerator/HelicalGenerator/data/bCov_4H_dataset/BCov_Models/'    





    
    

