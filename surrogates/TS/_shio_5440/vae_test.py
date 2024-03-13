#!/usr/bin/env python3
r"""
Predict test set

"""

import os
import numpy as np
import vae as vae
# Environment variable
try:
    epochs2use = int(os.environ['epochs2use'])
    kfold2use = int(os.environ['kfold2use'])
    print('epochs2use:', epochs2use, 'kfold2use:', kfold2use)
except:
    raise Exception("*** Must first set environment variable")

if __name__ == "__main__":
     
     # save obs / prediction to _output in ascii readable format
     for epoch in [epochs2use]: # predict using model trained up to # epochs
        print("epoch:", epoch)
 
        # load model, use cpu device (gpu performance not needed prediction)
        AE = vae.VarAutoEncoder()
        AE.load_model(model_name='vae_riku', device='cuda',kfold=kfold2use)

        # predict etamax
        AE.predict_dataset(epoch,kfold=kfold2use) #actual model prediction
 
        fname = '_output/vae_riku_test_k{:d}_{:04d}.npy'.format(kfold2use,epoch)
        pred_all = np.load(fname)

        fname = '_data/riku.npy'
        data = np.load(fname)

        fname = '_data/riku_test_index_k{:d}.txt'.format(kfold2use)
        test_index = np.loadtxt(fname).astype(int)
        obs_test = data[test_index, ...].max(axis=-1)

        ntest = len(test_index)

        pred_all = pred_all.max(axis=-1).mean(axis=0)
        pred_test = pred_all[-ntest:, :] # in the batch design, the last batch is the test set

        fname = '_output/etamax_VAE_predict_k{:d}_{:04d}.txt'.format(kfold2use,epoch)
        np.savetxt(fname, pred_test)

        fname = '_output/etamax_obs.txt'
        np.savetxt(fname, obs_test)




