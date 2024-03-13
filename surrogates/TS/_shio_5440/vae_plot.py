#!/usr/bin/env python3
r"""
Plot autoencoder prediction results

"""
import os, sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from aeplot import *
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import preprocessing
from pickle import load

# Environment variable
try:
    epochs2use = int(os.environ['epochs2use'])
    nepochs2use = int(os.environ['nepochs2use'])
    zdim2use = int(os.environ['zdim2use'])
    vae2use = int(os.environ['vae2use'])
    kfold2use = int(os.environ['kfold2use'])
    print('nepochs:', nepochs2use, 'zdim2use:', zdim2use,'vae2use:', vae2use, 'kfold2use:', kfold2use)
except:
    raise Exception("*** Must first set environment variable")


if __name__ == "__main__":

    # load dataset prediction
    model_name = 'vae_riku'
    epochs_lists = [epochs2use]
    nepochs = nepochs2use
    vaeruns = vae2use

    t_unif = np.linspace(0.0, 4.0 * 60, 1024) #data points to recreate time series
    gauges = [5010,5015,5020,5440]
    plot_history = True
    # hist_name_list = ['FUJI2011_42','NANKAI2022','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD',
    #     'SatakeMiniLowerSoft','SatakeMiniUpper','SatakeMiniUpperSoft','SatakeMiniLower']
    hist_name_list = ['FUJI2011_42','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD']
    fmt = 'png'
    dpi = 300

    if not os.path.exists('_plots'):
        os.mkdir('_plots')
  
    gauge = gauges[-1]
    for epochs_list in epochs_lists:

        print('processing for input time window and epoche ', epochs_list)
        ##
        ## Prediction vs. Observation plot
        ##
        
        #load obs data, pred results
        pred_train, obs_train, pred_test, obs_test, train_runno, test_runno \
            = load_obspred(model_name, kfold2use, epochs_list)

        epoch = epochs_list

        # create plot
        fig, ax, legends = plot_obsvpred(pred_train, obs_train,
                                    pred_test, obs_test)

        if plot_history:
            # add hist results
            hist_legends = legends
            markers = ['p', '*', '^','o', 's', 'D', 'v', '8', 'P', '>', '<']
            pred=[]
            obs=[]

            for ii, hist_name in enumerate(hist_name_list):
                # minutes = int(t_unif[T-1] * 60)
                fname = '_output/vae_riku_test_{:s}_k{:d}_e{:04d}.npy'.format(hist_name,kfold2use,epoch)
                pred_hist = np.load(fname)

                etamax_pred_hist = pred_hist.max(axis=-1).mean(axis=0)
                etamax_2std_hist = 2.0 * pred_hist.max(axis=-1).std(axis=0)

                fname = '_output/etamax_obs_{:s}.txt'.format(hist_name)
                etamax_obs_hist = np.loadtxt(fname)
            
                pred.append(etamax_pred_hist)
                obs.append(etamax_obs_hist[-1])

                line0, = ax.plot(etamax_obs_hist[-1],
                                    etamax_pred_hist,
                                    linewidth=0,
                                    marker=markers[ii],
                                    markersize=5,
                                    color='mediumblue',
                                    zorder=10)

                ax.errorbar(etamax_obs_hist[-1],
                            etamax_pred_hist,
                            yerr=etamax_2std_hist,
                            fmt='.',
                            color='mediumblue',
                            elinewidth=0.8,
                            alpha=0.3,
                            capsize=0,
                            markersize=2)

                # ax.set(xlim=(0, 13), ylim=(0, 13))

                hist_legends[0].append(line0)
                hist_legends[1].append(hist_name)

            ax.legend(hist_legends[0], hist_legends[1], fontsize=6, loc='upper left')

            ax.text(10, 4.5, '$R^2 Historic$:' + str(round(r2_score(obs, pred), 3)), fontsize=8)

        # set title / save figures
        title = "gauge {:3d}, fold {:d} " .format(gauge, kfold2use)

        ax.set_title(title, fontsize=10)

        fname = r'{:s}_predvobs_g{:03d}_k{:d}_e{:04d}.{:s}' \
            .format('vae', gauge, kfold2use, epoch, fmt)
        save_fname = os.path.join('_plots', fname)

        sys.stdout.write('\nsaving fig to {:s}'.format(save_fname))
        sys.stdout.flush()
        fig.tight_layout()
        fig.savefig(save_fname, dpi=300)
        # clean up
        plt.close(fig)
        
        
        ## Time-series plots for historic runs
        if plot_history: 
            for hist_name in hist_name_list:

                fname = 'vae_riku_obs_{:s}.npy'.format(hist_name)
                load_fname = os.path.join('_output', fname)

                obs_hist = (np.load(load_fname).squeeze())       
                pred2_hist = []
                
                epoch = epochs_list
                fname = 'vae_riku_test_{:s}_k{:d}_e{:04d}.npy'.format(hist_name, kfold2use,epoch)
                load_fname = os.path.join('_output', fname)
                pred_hist = np.load(load_fname)
                pred2_hist.append(pred_hist)  # TODO fixed indexing
            
                pred2_hist = (np.array(pred2_hist).squeeze())
                
                fig, axes = plot_pred_hist_timeseries(pred2_hist, obs_hist, gauges,
                                                    scale=1.2, dpi=300)

                title = '{:s}'.format(hist_name)
                axes[0].set_title(title)

                fig.tight_layout()
                fname = 'vae_timeseries_{:s}_k{:d}_e{:04d}.{:s}'.format(hist_name,kfold2use,epoch,fmt)
                fname = os.path.join('_plots', fname)
                fig.savefig(fname, dpi=dpi)

        #
        # Time-series plots for runs in the test set
        #
        for runno in test_runno[-5:]:
            pred2_run = pred_test[ :, test_runno == runno, :, :].squeeze()

            obs_run = obs_test[test_runno == runno, :, :].squeeze()

            fig, axes = plot_pred_timeseries(pred2_run, obs_run, gauges,
                                             scale=1.2, dpi=300)


            title = 'Realization #{:04d}, kfold: {:d} mins,epoch: {:04d}' \
                .format(runno, kfold2use,epoch)
            axes[0].set_title(title)

            fig.tight_layout()
            fname = 'vae_timeseries_test_r{:04d}_k{:d}_e{:04d}.{:s}'.format(runno,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)
            fig.savefig(fname, dpi=dpi)
            plt.close(fig)
       

        

       