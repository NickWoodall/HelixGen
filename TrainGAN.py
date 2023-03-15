import tensorflow as tf
import numpy as np
import itertools

import os
import pickle

import random

from sklearn.preprocessing import MinMaxScaler

import time
import joblib

import argparse
import textwrap


if tf.config.list_physical_devices('GPU'):
    device_name = tf.test.gpu_device_name()
    print(f'Device GPU availabe: {device_name}')
else:
    device_name = 'cpu'
    print(f'Only CPU Device available: {device_name}')


def prepData(name,perData=100):
    #prepare transformed Helix Fit Parameters for GAN from FitTransform
    
    rr = np.load(f'{name}', allow_pickle=True)
    X_train, y_train, featnames = [rr[f] for f in rr.files]

    #remove phi1
    X_train = X_train[:,:-4]
    if perData < 100:
        newSize = int(perData/100*len(X_train))

        X_train, y_train = zip(*random.sample(list(zip(X_train, y_train)), newSize))
        y_train = list(y_train)
    mm = MinMaxScaler(feature_range=(-1,1))
    X_train = mm.fit_transform(X_train)
    
    return X_train, y_train, mm


def make_generator_network(num_hidden_layers = 3,num_hidden_units = 64,num_output_units =28):
    
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(units=num_hidden_units, use_bias=False))
        model.add(tf.keras.layers.LeakyReLU())
        
    model.add(tf.keras.layers.Dense(units=num_output_units,activation='tanh'))
    
    return model

def make_discriminator_network(num_hidden_layers=3, num_hidden_units=64, num_output_units =1,drop=0.1):
    
    model = tf.keras.Sequential()
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(units=num_hidden_units))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(rate=drop))
        
    model.add(tf.keras.layers.Dense(units=num_output_units,activation=None))
    
    return model

def saveNetwork_Scaler(genMod,disMod,scaler,name,direc='data/'):
    
    joblib.dump(scaler, f'{direc}{name}_mm.gz')
    genMod.save(f'{direc}{name}.h5',overwrite=True,include_optimizer=True, save_format='h5')
    disMod.save(f'{direc}{name}disc.h5',overwrite=True,include_optimizer=True, save_format='h5')
        
def loadTrainLoss(name,direc='data/'):
 
    with open(f'{direc}{name}.pkl','rb') as f:
        data = pickle.load(f)
    
    all_losses = data[0]
    dval_losses = data[1]
    
    return all_losses, dval_losses

def init_model(X_train, y_train, batch=64 ,zIn=12, g_layers=3, g_size=64, d_layers =3, d_size=64,
               zmode='uniform',discDrop=0.1, normalization=None):

    z_size = zIn
    mode_z = zmode
    gen_hidden_layers= g_layers
    gen_hidden_size = g_size
    disc_hidden_layers = d_layers
    disc_hidden_size = d_size
    batch_size = batch

    h4_feat =X_train.shape[1]

    ds_train = tf.data.Dataset.from_tensor_slices((tf.cast(X_train, tf.float32), y_train))
    ds_train = ds_train.shuffle(X_train.shape[0]) 
    ds_train = ds_train.batch(batch_size, drop_remainder=True)

    with tf.device(device_name):
        gen_model = make_generator_network(num_hidden_layers=gen_hidden_layers,
                                   num_hidden_units = gen_hidden_size,
                                   num_output_units = h4_feat)
        gen_model.build(input_shape=(None,z_size))
        disc_model = make_discriminator_network(num_hidden_layers=disc_hidden_layers,
                                       num_hidden_units =disc_hidden_size,drop=discDrop)
        disc_model.build(input_shape=(None, h4_feat))

    return  gen_model, disc_model, ds_train
    

def percentRun(trainSettings, percentUsed = 100,feature_name='data/endpoint_distancemap_phi',nameOut=None):
    X_train, y_train, mm = prepData(feature_name,perData=percentUsed)
    print(f'X_train shape: {X_train.shape}')
    
    if nameOut:
        nameRun = f'_{nameOut}'
    else:
        nameRun = f'_examples{X_train.shape[0]:.0f}'
    #base_folder = time.strftime('log/%y%b%d_%I%M%p', time.localtime())
    base_folder = 'log/GAN'
    base_folder = f'{base_folder}{nameRun}/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['checkpoints','loss']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)


    settings = trainSettings
    
    @tf.function   
    def train_step(input_z,input_real):
        ##Compute generator's loss
        with tf.GradientTape() as g_tape:
            g_output = gen_model(input_z)
            d_logits_fake = disc_model(g_output, training=True)
            labels_real = tf.ones_like(d_logits_fake)
            g_loss = loss_fn(y_true=labels_real, y_pred = d_logits_fake)

        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)

        ##Optimization: Apply the gradients    
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads,gen_model.trainable_variables))

        ##Compute the discriminators loss
        with tf.GradientTape() as d_tape:
            d_logits_real = disc_model(input_real, training=True)
            d_labels_real = tf.ones_like(d_logits_real)

            d_loss_real = loss_fn(y_true=d_labels_real, y_pred = d_logits_real)

            d_logits_fake = disc_model(g_output, training=True)
            d_labels_fake = tf.zeros_like(d_logits_fake)

            d_loss_fake = loss_fn(y_true=d_labels_fake, y_pred=d_logits_fake)
            d_loss = d_loss_real + d_loss_fake

        ##compute the gradients of d_loss
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)

        ## Optimization : Apply the gradients
        d_optimizer.apply_gradients(grads_and_vars=zip(d_grads,disc_model.trainable_variables))
        d_probs_real = tf.reduce_mean(tf.sigmoid(d_logits_real))
        d_probs_fake = tf.reduce_mean(tf.sigmoid(d_logits_fake))

        return g_loss, d_loss, d_loss_real, d_loss_fake, d_probs_real, d_probs_fake



    def train(ds_train, epochs, manager, batch_size, z_size):

        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        start_time = time.time()
        for epoch in range(epochs):

            epoch_losses, epoch_d_vals = [],[]

            for i, (input_real,name) in enumerate(ds_train):
                input_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1, maxval=1)
                g_loss, d_loss, d_loss_real, d_loss_fake, d_probs_real, d_probs_fake = train_step(input_z,input_real)

                epoch_losses.append(( g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy()))
                epoch_d_vals.append((d_probs_real.numpy(), d_probs_fake.numpy()))


            checkpoint.step.assign_add(1)
            if (epoch + 1) % 25 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))


            all_losses.append(epoch_losses)
            all_d_vals.append(epoch_d_vals)

            track = f'Epoch {epoch:03d} |  ET {(time.time()-start_time)/60:.2f} min AvgLosses >> G/D '
            track = f'{track}{(np.mean(all_losses[-1][0],axis=0)):.3f}/{(np.mean(all_losses[-1][1],axis=0)):.3f}'
            track = f'{track} D Real :{(np.mean(all_losses[-1][2],axis=0)):.3f}'
            track = f'{track} D Fake :{(np.mean(all_losses[-1][3],axis=0)):.3f}'

            print(track)


    gen_model, disc_model, ds_train = init_model(X_train ,y_train, batch=settings['batch'], zIn=settings['input_size'], 
                                                 g_layers = settings['gen_layers'], g_size=settings['gen_size'], 
                                                 d_layers = settings['disc_layers'],d_size=settings['disc_size'],
                                                 zmode=settings['input_dist'], discDrop=settings['disc_dropOut'])

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    g_optimizer = tf.keras.optimizers.Adam(settings['Adam_gen_rate'])
    d_optimizer = tf.keras.optimizers.Adam(settings['Adam_disc_rate'])

    checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,discriminator_optimizer=d_optimizer,
                                 generator=gen_model,discriminator=disc_model,step=tf.Variable(1))

    cpD = settings['CheckPointDirec']
    checkpointDirec = f'{base_folder}{cpD}'

    manager = tf.train.CheckpointManager(checkpoint,checkpointDirec, max_to_keep=settings['CheckPointsKeep'])
    epoch_losses = []
    all_losses = []
    all_d_vals = []
    train(ds_train,settings['epochs'],manager,settings['batch'],settings['input_size'])

    outName = f'{nameRun[1:]}'
    saveNetwork_Scaler(gen_model, disc_model, mm, outName, direc=f'{base_folder}')
    
    

if __name__ == "__main__":
    
    settings = {'batch':4,
            'input_size':12,
            'input_dist':'uniform',
            'gen_layers':3,
            'gen_size':64,
            'disc_layers':3,
            'disc_size':64,
            'disc_dropOut':0.1,
            'Adam_gen_rate':.0001,
            'Adam_disc_rate':.0001,
            'CheckPointDirec':'checkpoints',
            'CheckPointsKeep':10,
            'epochs':300}

    all_losses = []
    all_d_vals = []


    parser = argparse.ArgumentParser(description="Train GAN. Additional settings at end of python file.")

    parser.add_argument("feature_data", help="Distance Map of 4 Helix Endpoints with 4 phi values at end as .npz", default='data/endpoint_distancemap_phi')
    parser.add_argument("-o","--out_name", help="Save name for log file. Defaults to number of examples" )
    parser.add_argument("-p","--percent", help="train on a random subset of data. Enter percent desired.", default = 100, type=int)
    parser.add_argument("-e","--epochs", help="number of epochs to train", default = 300, type=int)
    parser.add_argument("-d","--device_cpu", help="Use CPU to Train", action="store_true")
    
    
    

    args = parser.parse_args()
    
    settings['epochs'] = args.epochs
    if args.device_cpu:
        device_name = 'cpu'
        
    print(f'Using {device_name} to train')

    
    if args.percent < 100:
        percentRun(settings, percentUsed = args.percent,feature_name=args.feature_data, nameOut=args.out_name)
    else:
        percentRun(settings, percentUsed = 100,feature_name=args.feature_data, nameOut=args.out_name)






