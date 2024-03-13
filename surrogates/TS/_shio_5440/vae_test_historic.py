#!/usr/bin/env python3
r"""
Predict L1 test scenario: a special scenario generated from a different
earthquake generation method

"""

import os
import pandas as pd
import numpy as np
import vae as vae
from sklearn import preprocessing
from pickle import load

# Environment variable
try:
    epochs2use = int(os.environ['epochs2use'])
    kfold2use = int(os.environ['kfold2use'])
    print('epochs2use:', epochs2use, 'kfold2use:', kfold2use)
except:
    raise Exception("*** Must first set environment variable")

if __name__ == "__main__":
    # predict using model trained up to 1500 epochs
    for epoch in [epochs2use]: # predict using model trained up to # epochs
        print('epoch:',epoch)
        # load model, use cpu device (gpu performance is not needed for prediction)
        AE = vae.VarAutoEncoder()
        AE.load_model(model_name='vae_riku', device='cuda',kfold=kfold2use) #loads model and data

        # interpolate L1 gauge results
        gauges = AE.gauges
        ngauges = AE.ngauges
        print('gauges:',gauges)

        # hist_name_list = ['FUJI2011_42']
        # hist_name_list = ['SL_' + "{0:04}".format(n) for n in range(14)]
        # hist_name_list = ['FUJI2011_42','NANKAI2022','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD',
        #     'SatakeMiniLowerSoft','SatakeMiniUpper','SatakeMiniUpperSoft','SatakeMiniLower']
        hist_name_list = ['FUJI2011_42','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD']
        for hist_name in hist_name_list:
            print(hist_name)
            data_path = os.path.join('/mnt/beegfs/nragu/tsunami/japan/historic', hist_name)
            gauge_input = np.zeros((1, ngauges, 1024)) #data points to recreate time series
            etamax_obs = np.zeros((1, ngauges))
            etamax_pred = np.zeros((1, ngauges))

            for k, gauge in enumerate(gauges):
                fname = 'gauge{:05d}.txt'.format(gauge)
                load_fname = os.path.join(data_path, fname)
                # print(load_fname)
                raw_gauge = np.loadtxt(load_fname, skiprows=3)
                
                if hist_name == 'YAMAZAKI2018_TPMOD':
                    dz=raw_gauge[150,5]
                else:
                    dz=raw_gauge[1,5]               

                t = raw_gauge[:, 1]

                eta = raw_gauge[:, 5]-dz
                if hist_name == 'YAMAZAKI2018_TPMOD':
                    #set initial data to zero
                    eta[0:150]=0
                else:
                    eta[0:2]=0
                etamax_obs[0, k] = eta.max()
                
                # dz=raw_gauge[1,5]
                # t = raw_gauge[:, 1]
                # eta = raw_gauge[:, 5]-dz
                # etamax_obs[0, k] = eta.max()
                if k == 0:
                    # set prediction time-window based on threshold value of 0.1m
                    t_init = np.min(t[np.abs(eta) > 0.1])
                    t_final = t_init + 4.0 * 3600.0
                    t_unif = np.linspace(t_init, t_final, 1024) #data points to recreate time series
                
                # interpolate to uniform grid on prediction window
                gauge_input[0, k, :] = np.interp(t_unif, t, eta)

            pred_all = AE.predict_historic(gauge_input, epoch,kfold2use)
            pred_all = np.array(pred_all) # for one epoch, one kfold and one historic scenario

            # save predicted time-series
            save_fname = '_output/vae_riku_test_{:s}_k{:d}_e{:04d}.npy'.format(hist_name, kfold2use,epoch)
            np.save(save_fname, pred_all)

            # save interpolated time-series version of observations
            save_fname = '_output/vae_riku_obs_{:s}.npy'.format(hist_name)
            np.save(save_fname, gauge_input)

            # save true etamax from raw data
            save_fname = '_output/etamax_obs_{:s}.txt'.format(hist_name)
            np.savetxt(save_fname, etamax_obs)

            # save mean predicted etamax
            fname = '_output/etamax_VAE_predict_{:s}_k{:d}_e{:04d}.txt'.format(hist_name, kfold2use,epoch)
            etamax_pred = pred_all.squeeze().max(axis=-1)
            np.savetxt(fname, etamax_pred)
