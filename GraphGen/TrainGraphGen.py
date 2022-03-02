
from __future__ import print_function

import os, time, gzip
from collections import defaultdict
import urllib.request, json 
#import requests as reqs
import pickle


import json, time, os, sys, glob
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import shutil

# Library code
sys.path.insert(0,'struct2seq/')
sys.path.insert(0,f'experiments/')
from struct2seq import *
from utils import *
import data
import noam_opt

import util.AA_Exchange as aa
import argparse

from pymol import cmd, stored, selector
from weakref import WeakKeyDictionary
import os
from shutil import copyfile


from PDBDataset import StructureDataset_PDBDirec

def setup_model(hyperparams, device):
    # Build the model
    if hyperparams['model_type'] == 'structure':
        model = struct2seq.Struct2Seq(
            num_letters=hyperparams['vocab_size'], 
            node_features=hyperparams['hidden'],
            edge_features=hyperparams['hidden'], 
            hidden_dim=hyperparams['hidden'],
            k_neighbors=hyperparams['k_neighbors'],
            protein_features=hyperparams['features'],
            dropout=hyperparams['dropout'],
            use_mpnn=hyperparams['mpnn']
        ).to(device)
    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    return model
def setup_device_rng_new(argDict):
    # Set the random seed manually for reproducibility.
    np.random.seed(argDict['seed'])
    torch.manual_seed(argDict['seed'])
    # CUDA device handling.
    if torch.cuda.is_available():
        if not argDict['cuda']:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(argDict['seed'])
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    return device

def loss_smoothed_aa_weight(S_onehot, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def load_data(defDict,filename='data/refSet'):
    
    with open(f'{filename}_train.pkl', 'rb') as f:
        trainSet = pickle.load(f)
    with open(f'{filename}_test.pkl', 'rb') as f:
        testSet = pickle.load(f)


    print(f'test length is : {len(testSet.data)}')
    print(f'train length is : {len(trainSet.data)}')

    # Split the dataset
    lengthData = len(trainSet)
    trainSize = int(lengthData*.9)
    validSize = lengthData - trainSize
    dataset_indices = {d['name']:i for i,d in enumerate(trainSet)}
    train_set, validation_set = torch.utils.data.random_split(trainSet,(trainSize,validSize))

    #create dataset for mL algorithm 
    loader_train, loader_validation, loader_test = [data.StructureLoader(d, batch_size=defDict['batch_tokens']) 
                                                    for d in [train_set, validation_set, testSet]]
    print('Training:{}, Validation:{}, Test:{}'.format(len(train_set),len(validation_set),len(testSet)))
    
    return loader_train, loader_validation, loader_test
    
    
#default setting for making model


if __name__ == "__main__":

    defDict = {"hidden": 64, "k_neighbors": 30, "vocab_size": 20, "features": "full", "model_type": "structure", "mpnn": True, "restore": "", "name": "h128_full", "file_data": "data/chain_set.jsonl", "file_splits": "data/chain_set_splits.json", "batch_tokens": 1500, "epochs": 100, "seed": 1111, "cuda": True, "augment": False, "shuffle": 0.0, "dropout": 0.1, "smoothing": 0.05}
    
    parser = argparse.ArgumentParser(description="Saves Graph Gen Network to log file via timestamp")
    parser.add_argument("-i", "--inFile", help="Pickle Saved Test and Train Set Name", default='data/refSet')
    args = parser.parse_args()

    loader_train, loader_validation, loader_test = load_data(defDict,filename=args.inFile)



    #initiate model, device and checkpoint folders
    device = setup_device_rng_new(defDict)
    model = setup_model(defDict,device)
    optimizer = noam_opt.get_std_opt(model.parameters(), defDict['hidden'])
    criterion = torch.nn.NLLLoss(reduction='none')


    base_folder = time.strftime('log/%y%b%d_%I%M%p/', time.localtime())
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['checkpoints', 'plots']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    # Log files
    logfile = base_folder + 'log.txt'
    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain\tValidation\n')

    #train model using validation set to determine stopping.
    #load best checkpoint and 
    start_train = time.time()
    epoch_losses_train, epoch_losses_valid = [], []
    epoch_checkpoints = []
    total_step = 0
    for e in range(defDict['epochs']):
        # Training epoch
        model.train()
        train_sum, train_weights = 0., 0.
        for train_i, batch in enumerate(loader_train):

            start_batch = time.time()
            # Get a batch
            X, S, mask, lengths = featurize(batch, device, shuffle_fraction=defDict['shuffle'])
            elapsed_featurize = time.time() - start_batch

            optimizer.zero_grad()
            log_probs = model(X, S, lengths, mask)

            S_onehot = torch.nn.functional.one_hot(S,num_classes=20).float()

            _, loss_av_smoothed = loss_smoothed_aa_weight(S_onehot, log_probs, mask, weight=defDict['smoothing'])
            loss_av_smoothed.backward()
            optimizer.step()

            loss, loss_av = loss_nll(S, log_probs, mask)

            # Timing
            elapsed_batch = time.time() - start_batch
            elapsed_train = time.time() - start_train
            total_step += 1
            #print(total_step, elapsed_train, np.exp(loss_av.cpu().data.numpy()), np.exp(loss_av_smoothed.cpu().data.numpy()))

            if False:
                # Test reproducibility
                log_probs_sequential = model.forward_sequential(X, S, lengths, mask)
                loss_sequential, loss_av_sequential = loss_nll(S, log_probs_sequential, mask)
                log_probs = model(X, S, lengths, mask)
                loss, loss_av = loss_nll(S, log_probs, mask)
                print(loss_av, loss_av_sequential)

            # Accumulate true loss
            train_sum += torch.sum(loss * mask).cpu().data.numpy()
            train_weights += torch.sum(mask).cpu().data.numpy()

            # DEBUG UTILIZATION Stats
            #if defDict['cuda']:
            if False:
                utilize_mask = 100. * mask.sum().cpu().data.numpy() / float(mask.numel())
                utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
                tps_train = mask.cpu().data.numpy().sum() / elapsed_batch
                tps_features = mask.cpu().data.numpy().sum() / elapsed_featurize
                print('Tokens/s (train): {:.2f}, Tokens/s (features): {:.2f}, Mask efficiency: {:.2f}, GPU max allocated: {:.2f}'.format(tps_train, tps_features, utilize_mask, utilize_gpu))

            if total_step % 5000 == 0:
                print('epoch', e)
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict()
                }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step))

        # Train image
        plot_log_probs(log_probs, total_step, folder='{}plots/train_{}_'.format(base_folder, batch[0]['name']))

        # Validation epoch
        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            for _, batch in enumerate(loader_validation):
                X, S, mask, lengths = featurize(batch, device, shuffle_fraction=defDict['shuffle'])
                log_probs = model(X, S, lengths, mask)
                loss, loss_av = loss_nll(S, log_probs, mask)

                # Accumulate
                validation_sum += torch.sum(loss * mask).cpu().data.numpy()
                validation_weights += torch.sum(mask).cpu().data.numpy()

        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_perplexity = np.exp(validation_loss)
        print('Perplexity\tTrain:{}\t\tValidation:{}'.format(train_perplexity, validation_perplexity))

        # Validation image
        plot_log_probs(log_probs, total_step, folder='{}plots/valid_{}_'.format(base_folder, batch[0]['name']))

        with open(logfile, 'a') as f:
            f.write('{}\t{}\t{}\n'.format(e, train_perplexity, validation_perplexity))

        # Save the model
        checkpoint_filename = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step)
        torch.save({
            'epoch': e,
            'hyperparams': defDict,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict()
        }, checkpoint_filename)

        epoch_losses_valid.append(validation_perplexity)
        epoch_losses_train.append(train_perplexity)
        epoch_checkpoints.append(checkpoint_filename)

    # Determine best model via early stopping on validation
    best_model_idx = np.argmin(epoch_losses_valid).item()
    best_checkpoint = epoch_checkpoints[best_model_idx]
    train_perplexity = epoch_losses_train[best_model_idx]
    validation_perplexity = epoch_losses_valid[best_model_idx]
    best_checkpoint_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_model_idx + 1)
    shutil.copy(best_checkpoint, best_checkpoint_copy)
    load_checkpoint(best_checkpoint_copy, model)


    # Test epoch
    model.eval()
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        for _, batch in enumerate(loader_test):
            X, S, mask, lengths = featurize(batch, device)
            log_probs = model(X, S, lengths, mask)
            loss, loss_av = loss_nll(S, log_probs, mask)
            # Accumulate
            test_sum += torch.sum(loss * mask).cpu().data.numpy()
            test_weights += torch.sum(mask).cpu().data.numpy()

    test_loss = test_sum / test_weights
    test_perplexity = np.exp(test_loss)
    print('Perplexity\tTest:{}'.format(test_perplexity))

    with open(base_folder + 'results.txt', 'w') as f:
        f.write('Best epoch: {}\nPerplexities:\n\tTrain: {}\n\tValidation: {}\n\tTest: {}'.format(
            best_model_idx+1, train_perplexity, validation_perplexity, test_perplexity
        ))





