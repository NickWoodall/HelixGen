import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

import random
import pickle

from sklearn.preprocessing import MinMaxScaler

import time
#from pickle import dump
import joblib
import argparse


def make_plot_orig(loss, d_vals, save_name=None):
    """Plot Generator and Discriminator losses"""
    epoch = len(loss)
    
    fig = plt.figure(figsize=(16,6))
    ##plotting the losses
    ax = fig.add_subplot(1,2,1)

    g_losses = [item[0] for item in itertools.chain(*loss)]
    d_losses = [item[1]/2.0 for item in itertools.chain(*loss)]

    plt.plot(g_losses, label='Generator loss', alpha=0.95)
    plt.plot(d_losses, label='Discriminator loss', alpha=0.95)
    plt.legend(fontsize=20)

    ax.set_xlabel('Iteration', size=15)
    ax.set_ylabel('Loss', size=15)

    epoch2iter = lambda e: e*len(loss[-1])
    epoch_ticks = list(range(0,epoch+1,25))
    epoch_ticks[0] = 1
    newpos = [epoch2iter(e) for e in epoch_ticks]
    ax2 = ax.twiny()
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(epoch_ticks)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward',60))
    ax2.set_xlabel('Epoch', size=15)
    ax.set_xlim(ax.get_xlim())
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    ##Plotting outputs of discriminator

    ax = fig.add_subplot(1,2,2)
    d_vals_real = [item[0] for item in itertools.chain(*d_vals)]
    d_vals_fake = [item[1] for item in itertools.chain(*d_vals)]
    plt.plot(d_vals_real, alpha=0.75, label='Real $D(\mathbf{x})$')
    plt.plot(d_vals_fake, alpha=0.75, label='Fake $D(\mathbf{x})$')
    plt.legend(fontsize=20)


    ax.set_xlabel('Iteration', size=15)
    ax.set_ylabel('Discriminator', size=15)


    ax2 = ax.twiny()
    ax2.set_xticks(newpos)
    ax2.set_xticklabels(epoch_ticks)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward',60))
    ax2.set_xlabel('Epoch', size=15)
    ax.set_xlim(ax.get_xlim())
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    
    if save_name is not None:
        fig.savefig(save_name)
        
        
def loadTrainLoss(name):
 
    with open(name,'rb') as f:
        data = pickle.load(f)
    
    all_losses = data[0]
    dval_losses = data[1]
    
    return all_losses, dval_losses


def load_and_save(load_name, save_name):
    all_losses, all_d_vals = loadTrainLoss(load_name)
    make_plot_orig(all_losses, all_d_vals,save_name=save_name)
    
    
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Plot and save Generator and Discriminator Losses")

    parser.add_argument("-i","--inFile", help="Pickled loss data input data load.")
    parser.add_argument("-o","--outFile", help="Name of to save plot as png.", default = 'loss_plot.png')

    args = parser.parse_args()
    
    load_and_save(args.inFile,args.outFile)
    
    
    
    


