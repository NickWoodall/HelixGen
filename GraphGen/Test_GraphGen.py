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

def get_test_names(clusterLoad = '../Clustering/data/refData.npz',test=True):

    #hard coded from manually expecting clusters organized by AlignCluster
    #describes whether termini are together or apart, 
    #Approximately 15122 in together, 11556 in apart
    #seperate train and test class
    together=[0,1,2,8,9,14,15,17,18,19,20,23]
    apart = [3,4,5,6,7,10,11,12,13,16,24,25]

    if test:
        clusNums = apart
    else:
        clusNums = together

    with np.load(f'{clusterLoad}', allow_pickle=True) as rr:
        y_, y_train, X_train, featNames = [rr[f] for f in rr.files]
    cluster_labels = np.unique(y_)
    n_clusters = cluster_labels.shape[0]
    
    nameList = []
    
    for x in clusNums:
        nameList.extend(y_train[np.where(y_==x)])
        
    return nameList


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
            seq = f'{seq}{aa.aaCodeExchange(x)}'

        seqActual.append(seq)

    return seqActual

def compare_true_predicted(seqPred,seqActual,norm=False):
    
    alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    si = len(alphabet)

    matrix = np.zeros((si,si))
    #Confusion matrix whose i-th row and j-th column entry indicates the number of samples 
    #with true label being i-th class and predicted label being j-th class.
    for x in range(len(seqActual)):

        predicted = list(seqPred[x][0])
        actual = list(seqActual[x])

        mc = confusion_matrix(actual,predicted, labels=alphabet)

        matrix += mc
        
    if norm:
        normDiv = matrix.max(axis=0)
        inde=normDiv<1
        normDiv[inde] = 1
        return (matrix / normDiv[:,None])*100
    
    return matrix


def graphmodel_stats(struct_direc,seqPDB_direc, chkpath='data/best_checkpoint_epoch71.pt',limit = 5,hidden=64):

    sP = ps.getSeq(chkpath,struct_direc,test=True,limit=limit,hidden=hidden)
    
    sA = get_pdb_seq(sP,direcName=seqPDB_direc)
        
    matrix = compare_true_predicted(sP,sA)  
    
    return sP, sA, matrix
        
def percentAccurate(confusion_matrix, group = None):
    """Returns the indvidual percent accuracy for each amino acid in the confusion matrix"""
                    
    
    diag = np.array([x for x in range(20)])
    
    if group is not None:
        #ILV = np.array([7,9,17])
        diag = np.array([x for x in range(len(group))])
        
        groupMat = confusion_matrix[group]
        tot = groupMat.sum()
        correct = groupMat[diag,group].sum()
        
        return int(correct/tot*100)

    tot = confusion_matrix.sum(axis=1) #sum along column
    tot[tot<1] = 1
    accuracy = confusion_matrix[diag,diag]/tot*100
    
    
    return accuracy

def groupCorrect(confusion_matrix, group):
    """Assuming that an amino acid assignment in each group is equivalent, give accuracy with this metric"""
    
    tot = confusion_matrix[group].sum()
    
    #all possible indices from the group
    dex = np.array(np.meshgrid(group,group)).T.reshape(-1,2)
    
    correct = confusion_matrix[dex[:,0],dex[:,1]].sum()
    
    return int((correct/tot)*100)

#Confusion matrix whose i-th row and j-th column entry indicates the number of samples 
#with true label being i-th class and predicted label being j-th class.
# matrix = getStats(checkpoint_path, testProt = 100, norm=False)
# pa = percentAccurate(matrix).astype(np.int32)

def group_accuracy(matrix):
    AILV = np.array([0,7,9,17])
    AILVF = np.array([0,4,7,9,17])
    AVILFYW = np.array([0,4,7,9,10,17,18,19])
    DE_KR = np.array([2,3,8,14])
    PG = np.array([5,12])
    POLARS = np.array([2,3,6,8,11,13,14,15,16]) #D,E,H,K,N,Q,R,S,T
    # ref        0   1   2   3   4   5   6   7   8   9   10  11  12  13  14 15  16  17  18  19  
    alphabet = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    si = len(alphabet)
    
    pa = percentAccurate(matrix).astype(np.int32)
    phobic_a = percentAccurate(matrix, group=AILVF)
    charged_a = percentAccurate(matrix, group=DE_KR)
    groupCharge = groupCorrect(matrix,group=POLARS) 
    groupPho = groupCorrect(matrix,group=AILVF)
    print(f'GroupCharge {groupCharge}%')
    print(f'GroupPhobic {groupPho}%')
    print(f'Hydrophobic Accuracy AILVF {phobic_a}%')
    print(f'Charged Accuracy DE_RK {charged_a}%')
    print(f' AFILV {pa[AILVF]}')
    print(f' DE KR {pa[DE_KR]}' )
    print(f' G P {pa[PG]}')
    print('')
    
    
#------------ Code to save and analyze Rosetta Scores ------------------------

def save_scoreList(direc,nameList=[],scoreList=[]):

    with open(f'{direc}_nameList.pkl',"wb") as f:
        pickle.dump(nameList,f)

    with open(f'{direc}_scoreList.pkl',"wb") as f:
        pickle.dump(scoreList,f)

def openArray(direc):
    
    with open(f'{direc}/nameList.pkl', 'rb') as f:
        nameList = pickle.load(f)
    with open(f'{direc}/scoreList.pkl', 'rb') as f:
        scoreList = pickle.load(f)
        
    return nameList, scoreList

def extract_total_score(scoresList,regPlace=[],dictPlace=[2,5,8,10]):
    
    outList = []
    #get total score and normalize to sequence length, final score needs associated sequence in different array place
    for x in scoresList:
        inner = []
        for y in range(len(x)):
            if y in regPlace:
                inner.append(x[y])
            if y in dictPlace:
                if y == 10:
                    inner.append(x[y]['total_score']/len(x[7]))
                else:  
                    inner.append(x[y]['total_score']/len(x[y-1]))
        outList.append(inner)
        
    return outList

def extract_total_score2(scoresList,regPlace=[],dictPlace=[2,5,8,10]):
    
    outList = []
    #get total score and normalize to sequence length
    for x in scoresList:
        inner = []
        for y in range(len(x)):
            if y in regPlace:
                inner.append(x[y])
            if y in dictPlace:
                if y == 7:
                    inner.append(x[y]['total_score']/len(x[4]))
                else:  
                    inner.append(x[y]['total_score']/len(x[y-1]))
        outList.append(inner)
        
    return outList

def extract_stats(scoresList,scPlace=[2],scorePlace=[2]):
    
    outList = []
    
    for x in scoresList:
        inner = []
        for y in range(len(x)):
            if y in scPlace:
                inner.append(x[y]['sc'])
            if y in scorePlace:  
                inner.append(x[y]['total_score']/len(x[y-1]))
                
        inner.append(len(x[1]))
        inner.append(x[3])
        outList.append(inner)
        
    return outList
    
labels3=['name',#0
  'pred_seq_this_work','pred_scores_this_work','pred_time_this_work', #1,2,3
  'pred_seq_graphGen','pred_scores_graphGen','pred_time_graphGen', #4,5,6
  'actual_seq_graphGen','actual_scores_graphGen','actual_time_graphGen',#7,8,9
  'orig_scores'] #10
  
labels2=['name',#0
  'pred_seq_this_work','pred_scores_this_work','pred_time_this_work', #1,2,3
  'actual_seq','actual_scores','actual_time',#4,5,6
  'orig_scores'] #7
  
  
def plot_relative_scores3(nameList, scoreList,dictPlace=[2,5,8,10],outName='scores/testFig.png'):
    #get total score and normalize to sequence length
    sL = extract_total_score(scoreList,dictPlace=dictPlace)
    l2 = ['pred_str','pred_orig','actual','original']
    df = pd.DataFrame(sL,columns=l2)
    df = df.rename(columns={"pred_str":"PredictSeq 4H_GG","original": "reference set", "actual": "Reference Sequence:Relax", "pred_orig":"PredictSeqGG:Relax"})
    histp = sns.histplot(data=df)
    histp.figure.savefig(outName)
    
def plot_relative_scores2(nameList, scoreList,dictPlace=[2,5,7],outName='scores/testFig.png'):
    #get total score and normalize to sequence length
    sL = extract_total_score2(scoreList,dictPlace=dictPlace)
    l2 = ['pred_str','actual','original']
    df = pd.DataFrame(sL,columns=l2)
    df = df.rename(columns={"pred_str":"PredictSeq 4H_GG","original": "reference set", "actual": "Reference Sequence:Relax"})
    histp = sns.histplot(data=df)
    histp.figure.savefig(outName)

    
    

  
#------------ Code to call Rosetta methods for desiging proteins using the GraphGen Network -----------

def rescore(struct_direc,nameList=[],scoreList=[],outDirec=''):
    
    fL = os.listdir(struct_direc)
    counter = 0 
    
    for x in fL:
        
        if not x.endswith('.pdb'):
            continue
        
        name = x[:-4]#remove .pdb
        pose = pyrosetta.pose_from_file(f'{struct_direc}{x}')
        pose_score = ros.scoreOnly(pose)
        pose_score_sc = ros.score_sssc(pose_score)
        seq = pose_score_sc.sequence()
        
        score = pose_score_sc.scores['total_score']
        ss_sc = pose_score_sc.scores['sc']
        
        print(f'score is {score/len(seq):.2f}')
        print(f'sc is {ss_sc:.2f}')
                                        
        scoreList.append([name,seq,ros.get_pose_scores(pose_score_sc),15])
        nameList.append(name)
                                        
        if counter %10 == 0:
            save_scoreList(nameList=nameList,scoreList=scoreList,direc=outDirec)
                                        
        counter += 1
                                        
    save_scoreList(nameList=nameList,scoreList=scoreList,direc=outDirec)
                                        

def relax_stats2(sP_thiswork,seqActual,
                dOrig='../data/bCov_4H_dataset/BCov_Models/', dStraight = '../data/bCov_4H_dataset/BCov_Models_Straight/',
               scoreList=[],nameList=[], outDirec='scores'):
    
    labels = ['name',
          'pred_seq_this_work','pred_scores_this_work','pred_time_this_work',
          'actual_seq','actual_scores','actual_time',
          'orig_scores']

    
    
    for x in range(len(sP_thiswork)):
        pred_seq_thiswork = sP_thiswork[x][0]
        name = sP_thiswork[x][1]

        #change residues to graphGen prediction
        pose = pyrosetta.pose_from_file(f'{dStraight}/{name}_str.pdb')
        pred_tw_pose = ros.mutate_residue(pose,pred_seq_thiswork)
        pred_sa_pose = ros.mutate_residue(pose,seqActual[x])


        #relax using predicted sequence this work
        start_pred = time.time()
        pred_tw_pose_relax = ros.fastRelax(pred_tw_pose)
        end_pred = time.time()
        pred_tw_time_relax = end_pred - start_pred
        
        #relax using predicted sequence actual
        start = time.time()
        sa_pose_relax = ros.fastRelax(pred_sa_pose)
        end = time.time()
        sa_pose_time_relax = end - start
        
        #get score of original
        pose_orig = pyrosetta.pose_from_file(f'{dOrig}/{name}.pdb')
        pose_orig_score = ros.scoreOnly(pose_orig)

        scoreList.append([name,
                 pred_seq_thiswork, ros.get_pose_scores(pred_tw_pose_relax), pred_tw_time_relax,
                 seqActual[x],ros.get_pose_scores(sa_pose_relax),sa_pose_time_relax,ros.get_pose_scores(pose_orig_score)])
    
        nameList.append(name)
        
    save_scoreList(outDirec,nameList=nameList,scoreList=scoreList)
    
    return nameList, scoreList
               
               
def relax_stats3(sP_thiswork,sP_graphGenOrig,seqActual,
                dOrig='../data/bCov_4H_dataset/BCov_Models/', dStraight = '../data/bCov_4H_dataset/BCov_Models_Straight/',
               scoreList=[],nameList=[], outDirec='scores'):
    
    labels = ['name',
          'pred_seq_this_work','pred_scores_this_work','pred_time_this_work',
          'pred_seq_graphGen','pred_scores_graphGen','pred_time_graphGen',
          'actual_seq','actual_scores','actual_time','orig_scores']

    
    
    for x in range(len(sP_thiswork)):
        pred_seq_thiswork = sP_thiswork[x][0]
        pred_seq_graphGenOrig = sP_graphGenOrig[x][0]
        name = sP_thiswork[x][1]

        #change residues to graphGen prediction
        pose = pyrosetta.pose_from_file(f'{dStraight}/{name}_str.pdb')
        pred_gg_pose = ros.mutate_residue(pose,pred_seq_graphGenOrig)
        pred_tw_pose = ros.mutate_residue(pose,pred_seq_thiswork)
        pred_sa_pose = ros.mutate_residue(pose,seqActual[x])


        #relax using predicted sequence this work
        start_pred = time.time()
        pred_tw_pose_relax = ros.fastRelax(pred_tw_pose)
        end_pred = time.time()
        pred_tw_time_relax = end_pred - start_pred
        
        #relax using predicted sequence gg
        start_pred = time.time()
        pred_gg_pose_relax = ros.fastRelax(pred_gg_pose)
        end_pred = time.time()
        pred_gg_time_relax = end_pred - start_pred
        
        #relax using predicted sequence actual
        start = time.time()
        sa_pose_relax = ros.fastRelax(pred_sa_pose)
        end = time.time()
        sa_pose_time_relax = end - start
        
        #get score of original
        pose_orig = pyrosetta.pose_from_file(f'{dOrig}/{name}.pdb')
        pose_orig_score = ros.scoreOnly(pose_orig)

        scoreList.append([name,
                 pred_seq_thiswork, ros.get_pose_scores(pred_tw_pose_relax), pred_tw_time_relax,
                 pred_seq_graphGenOrig, ros.get_pose_scores(pred_gg_pose_relax), pred_gg_time_relax,
                 seqActual[x],ros.get_pose_scores(sa_pose_relax),sa_pose_time_relax,ros.get_pose_scores(pose_orig_score)])
    
        nameList.append(name)
        
    save_scoreList(outDirec,nameList=nameList,scoreList=scoreList)
    return nameList, scoreList

modelDirec = '../data/4H_dataset/models/'
straightDirec = '../data/4H_dataset/str_models/'


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test the GraphGen network.")
    
    parser.add_argument("-i", "--inDirec", help="Input Structure Directory",default=straightDirec)
    parser.add_argument("-s", "--seqDirec", help="Input Directory for True sequence in pdb file format.",default=modelDirec)
    
    parser.add_argument("-g", "--graphGen", help="Location of GraphGen Network", default='data/Best_GraphGen.pt')
    parser.add_argument("--graphGenOriginal", help="Location of Original Dataset GraphGen Network", default='')
    parser.add_argument("-l", "--limitInput", help="Limit the number of trials.", default=100, type=int)
    parser.add_argument("-o", "--output_name", help="Output name for Rosetta Scores. def: scores/test1", default='scores/test1')
    
    parser.add_argument("-t", "--testDataset", help="Analyze the AA predictions for the test dataset.", action="store_true")
    parser.add_argument("-r", "--trainDataset", help="Analyze the AA predictions for the test dataset.", action="store_true")
    parser.add_argument("--testOriginalGraphGen", help="In additaion, Analyze the AA predictions for the GraphGen trained on the original dataset.", action="store_true")
    
    parser.add_argument("--score", help="Analyze the AA predictions for the test dataset via Rosetta Score (Predict, Mutate, Relax).", action="store_true")
    
    args = parser.parse_args()
    
    
    pyrosetta.init("-beta -mute all")
    
    
    if args.score:

        if args.testDataset:
            sP = ps.getSeq(args.graphGen, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=64, mpnn=True,test=True, train=False,catSet=False)
            sA = get_pdb_seq(sP,direcName=args.seqDirec,removeEnding=False)
            if args.testOriginalGraphGen:
                sG = ps.getSeq(args.graphGenOriginal, direcName = args.seqDirec,dtest=args.inDirec, 
                   limit = args.limitInput, hidden=128, mpnn=True,test=True, train=False,catSet=False)
        elif args.trainDataset:
            sP = ps.getSeq(args.graphGen, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=64, mpnn=True,test=False, train=True,catSet=False)
            sA = get_pdb_seq(sP,direcName=args.seqDirec,removeEnding=False)
            if args.testOriginalGraphGen:
                sG = ps.getSeq(args.graphGenOriginal, direcName = args.seqDirec,dtest=args.inDirec, 
                   limit = args.limitInput, hidden=128, mpnn=True,test=False, train=True,catSet=False)
        else:
        
            sP = ps.getSeq(args.graphGen, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=64, mpnn=True,test=False, train=False,catSet=False)
            sA = get_pdb_seq(sP,direcName=args.seqDirec,removeEnding=False)
            if args.testOriginalGraphGen:
                sG = ps.getSeq(args.graphGenOriginal, direcName = args.seqDirec,dtest=args.inDirec, 
                   limit = args.limitInput, hidden=128, mpnn=True,test=False, train=False,catSet=False)
        
        print('Statistics for straight Helix trained GraphGen Network')
        matrixP = compare_true_predicted(sP,sA)
        group_accuracy(matrixP)
        
        if args.testOriginalGraphGen:
            print('Statistics for originally trained GraphGen Network')
            matrix = compare_true_predicted(sG,sA)
            group_accuracy(matrix)
        
        print('Mutating Structures to Predicted AAs, Relaxing and Scoring') 
        
        if args.testOriginalGraphGen:
            nameList, scoreList = relax_stats3(sP,sG,sA,dOrig=args.seqDirec, dStraight = args.inDirec, outDirec=args.output_name)
            plot_relative_scores3(nameList, scoreList,dictPlace=[2,5,8,10],outName=f'{args.output_name}_barplot.png')
        else:
            nameList, scoreList = relax_stats2(sP,sA,dOrig=args.seqDirec, dStraight = args.inDirec, outDirec=args.output_name)
            plot_relative_scores2(nameList, scoreList,dictPlace=[2,5,7],outName=f'{args.output_name}_barplot.png')
            
    
    
    
    
    elif args.testDataset:

        start_pred = time.time()
        
        #returns a sequence/ name of the protein
        sP = ps.getSeq(args.graphGen, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=64, mpnn=True,test=True, train=False,catSet=False)
        end_pred = time.time()
        
        predTime = end_pred - start_pred
        print(f'{len(sP)} test sequences predicted in {predTime:0.1f}s.')
        
        sA = get_pdb_seq(sP,direcName=args.seqDirec,removeEnding=False)
        print('Statistics for straight helix trained GraphGen')
        matrix = compare_true_predicted(sP,sA)
        group_accuracy(matrix)  #prints accuracy
        
        if args.testOriginalGraphGen:
            sG = ps.getSeq(args.graphGenOriginal, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=128, mpnn=True,test=True, train=False,catSet=False)
               
            print('Statistics for originally trained GraphGen Network')
            matrix = compare_true_predicted(sG,sA)
            group_accuracy(matrix)  #prints accuracy
        
    elif args.trainDataset:
    
        start_pred = time.time()
        
        #returns a sequence/ name of the protein
        sP = ps.getSeq(args.graphGen, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=64, mpnn=True,test=False, train=True,catSet=False)
        end_pred = time.time()
        
        predTime = end_pred - start_pred
        print(f'{len(sP)} train sequences predicted in {predTime}.')
        
        sA = get_pdb_seq(sP,direcName=args.seqDirec,removeEnding=False)
        
        matrix = compare_true_predicted(sP,sA)
        group_accuracy(matrix) #prints accuracy
        
        if args.testOriginalGraphGen:
            sG = ps.getSeq(args.graphGenOriginal, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=128, mpnn=True,test=False, train=True,catSet=False)
               
            print('Statistics for originally trained GraphGen Network')
            matrix = compare_true_predicted(sG,sA)
            group_accuracy(matrix)
            
    else:
    
        start_pred = time.time()
        
        #returns a sequence/ name of the protein
        sP = ps.getSeq(args.graphGen, direcName = args.seqDirec,dtest=args.inDirec, limit = args.limitInput, hidden=64, mpnn=True,test=False, train=False,catSet=False)
        end_pred = time.time()
        
        predTime = end_pred - start_pred
        print(f'{len(sP)} sequences predicted in {predTime:0.1f}s.')
        
        sA = get_pdb_seq(sP,direcName=args.seqDirec,removeEnding=False)
        
        matrix = compare_true_predicted(sP,sA)
        group_accuracy(matrix) #prints accuracy
        
        if args.testOriginalGraphGen:
            sG = ps.getSeq(args.graphGenOriginal, direcName = args.seqDirec,dtest=args.inDirec, 
               limit = args.limitInput, hidden=128, mpnn=True,test=False, train=True,catSet=False)
               
            print('Statistics for originally trained GraphGen Network')
            matrix = compare_true_predicted(sG,sA)
            group_accuracy(matrix)
