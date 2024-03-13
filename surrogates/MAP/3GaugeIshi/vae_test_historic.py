#!/usr/bin/env python3
r"""
Predict L1 test scenario: a special scenario generated from a different
earthquake generation method

"""

import os
import numpy as np
import vae as vae


# Environment variable
try:
    vae2use = int(os.environ['vae2use'])
    # epochs2use = int(os.environ['epochs2use'])
    epochs2use =  [int(epoch) for epoch in os.environ['epochs2use'].split(',')]
    kfold2use = int(os.environ['kfold2use'])
    print('epochs2use:', epochs2use, 'kfold2use:', kfold2use, 'vae2use:', vae2use)
except:
    raise Exception("*** Must first set environment variable")

if __name__ == "__main__":
    # predict using model trained up to 1500 epochs
    for epoch in epochs2use: # predict using model trained up to # epochs
        print('epoch:',epoch)
        # load model, use cpu device (gpu performance is not needed for prediction)
        AE = vae.VarAutoEncoder()
        AE.load_model(model_name='vae_riku', device='cuda',kfold=kfold2use) #loads model and data

        gauges = AE.gauges
        ninput = AE.ninput
        print('gauges:',gauges)

        # hist_name_list = ['FUJI2011_42']
        # hist_name_list = ['SL_' + "{0:04}".format(n) for n in range(14)]
        # hist_name_list = ['FUJI2011_42','NANKAI2022','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD',
        #     'SatakeMiniLowerSoft','SatakeMiniUpper','SatakeMiniUpperSoft','SatakeMiniLower']
        hist_name_list = ['FUJI2011_42','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD']
        hdx = [0,2,3,4,5]
        #load footprint historic data
        footinput = np.load('_data/riku_hisinputs.npy')[hdx,:,:]
        

        for h,hist_name in enumerate(hist_name_list):
            data_path = os.path.join('/mnt/beegfs/nragu/tsunami/japan/historic', hist_name)
            gauge_input = np.zeros((1, ninput, 1024)) #data points to recreate time series
            etamax_pred = np.zeros((1, ninput))

            for k, gauge in enumerate(gauges): #to use as list in enumerate function stop before footprint gauge
                fname = 'gauge{:05d}.txt'.format(gauge)
                load_fname = os.path.join(data_path, fname)
                raw_gauge = np.loadtxt(load_fname, skiprows=3)

                t = raw_gauge[:, 1]
                eta = raw_gauge[:, 5]
                if hist_name == 'YAMAZAKI2018_TPMOD':
                    print(hist_name)
                    dz=raw_gauge[90,5]
                    eta = eta-dz
                else:
                    dz=raw_gauge[1,5]
                    eta = eta-dz  
  
                if k == 0:
                    # set prediction time-window
                    t_init = np.min(t[np.abs(eta) > 0.1])
                    t_final = t_init + 4.0 * 3600.0
                    t_unif = np.linspace(t_init, t_final, 1024) #data points to recreate time series

                # interpolate to uniform grid on prediction window
                gauge_input[0, k, :] = np.interp(t_unif, t, eta)

            #calculate flood depths output        
            max_obs = np.array([eta.max(),footinput[h,0,:].max()]) #max of input and output from observations
            print(hist_name,max_obs)

            pred_all = AE.predict_historic(gauge_input, epoch,kfold2use,vae2use)
            pred_all = np.array(pred_all).squeeze() 
            print('pred_all.shape:',pred_all.shape)

            # save predicted flood depths
            save_fname = '_output/vae_riku_test_{:s}_k{:d}_e{:04d}.npy'.format(hist_name, kfold2use,epoch)
            np.save(save_fname, pred_all)

            # save interpolated time-series version of observations inputs 
            save_fname = '_output/vae_riku_obs_{:s}.npy'.format(hist_name)
            np.save(save_fname, gauge_input)

            # save true etamax from raw data( max wave height and flood depth from observations)
            save_fname = '_output/etamax_obs_{:s}.txt'.format(hist_name)
            np.savetxt(save_fname, max_obs)

            # save predicted mean flood depths( mean of 100 samples)
            fname = '_output/etamax_VAE_meanpredict_{:s}_e{:04d}.txt'.format(hist_name,epoch)
            etamax_pred = pred_all.mean(axis=0).squeeze()
            print('etamax_pred.shape:',etamax_pred.shape)
            np.savetxt(fname, etamax_pred)
