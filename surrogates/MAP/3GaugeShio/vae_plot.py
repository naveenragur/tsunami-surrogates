#!/usr/bin/env python3
r"""
Plot autoencoder prediction results

"""

import os
import sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
matplotlib.use('Agg')
import contextily as cx
import vae as vae
from sklearn.metrics import r2_score,mean_squared_error

# Environment variable
# Environment variable
try:
    #epochs2use is string of values separated by comma 0,12000,...convert to list of integers 
    epochs2use =  [int(epoch) for epoch in os.environ['epochs2use'].split(',')]
    # epochs2use = int(os.environ['epochs2use'])
    zdim2use = int(os.environ['zdim2use'])
    vae2use = int(os.environ['vae2use'])
    kfold2use = int(os.environ['kfold2use'])
    print('zdim2use:', zdim2use,'vae2use:', vae2use, 'kfold2use:', kfold2use)
except:
    raise Exception("*** Must first set environment variable")

def Gfit(obs, pred): #a normalized least-squares
    obs = np.array(obs)
    pred = np.array(pred)
    Gvalue = 1 - (2*np.sum(obs*pred)/(np.sum(obs**2)+np.sum(pred**2)))
    return Gvalue


if __name__ == "__main__":

    # load dataset prediction
    model_name = 'vae_riku'

    vaeruns = vae2use
    gauge = '5010,5015,5020' #gauges = [5832,6042] 
    plot_history = True
    # hist_name_list = ['FUJI2011_42','NANKAI2022','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD',
    #     'SatakeMiniLowerSoft','SatakeMiniUpper','SatakeMiniUpperSoft','SatakeMiniLower']
    hist_name_list = ['FUJI2011_42','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD']
    hdx = [0,2,3,4,5]
    fmt = 'png'
    dpi = 300

    if not os.path.exists('_plots'):
        os.mkdir('_plots')
   
    ##
    ## Prediction vs. Observation plot
    ##
    for epoch in epochs2use:
        #load obs data, pred results
        pred_train, obs_train, pred_test, obs_test, train_runno, test_runno \
            = vae.load_obspred(model_name, kfold2use, epoch)

        # # create plot
        fig_list = vae.plot_obsvpred(pred_train, obs_train,
                                    pred_test, obs_test)
        fig, ax, legends = fig_list[0]

        if plot_history:
            # add hist results
            hist_legends = legends
            markers = ['p', '*', '^','o', 's', 'D', 'v', '8', 'P', '>', '<']
            pred=[]
            obs=[]

            for ii, hist_name in enumerate(hist_name_list):
                print(epoch,hist_name)

                fname = '_output/vae_riku_test_{:s}_k{:d}_e{:04d}.npy'.format(hist_name,kfold2use,epoch)
                pred_hist = np.load(fname)

                etamax_pred_hist = pred_hist.max(axis=-1).mean(axis=0).squeeze()
                etamax_2std_hist = 2.0 * pred_hist.max(axis=-1).std(axis=0).squeeze()

                fname = '_output/etamax_obs_{:s}.txt'.format(hist_name)
                etamax_obs_hist = np.loadtxt(fname)[1] #max flood depth

                pred.append(etamax_pred_hist)
                obs.append(etamax_obs_hist)


                line0, = ax.plot(etamax_obs_hist,
                                etamax_pred_hist,
                                linewidth=0,
                                marker=markers[ii],
                                markersize=5,
                                color='mediumblue',
                                zorder=10)

                ax.errorbar(etamax_obs_hist,
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
        title = "Max flood depth gauge using input {:s}, fold {:d} ".format(gauge, kfold2use)

        ax.set_title(title, fontsize=10)

        fname = r'{:s}_predvobs_inputg{:s}_k{:d}_e{:04d}.{:s}' \
            .format('vae',gauge, kfold2use, epoch, fmt)
        save_fname = os.path.join('_plots', fname)

        sys.stdout.write('\nsaving fig to {:s}'.format(save_fname))
        sys.stdout.flush()
        fig.tight_layout()
        fig.savefig(save_fname, dpi=300)
        # clean up
        plt.close(fig)
        
        # Map plots for historic runs
        #
        fname = 'fgmax0001_flooded.csv'
        load_fname = os.path.join('_data', fname)
        fgmaxloc=np.loadtxt(load_fname, delimiter=',',usecols=(0,1,2))

        fname = 'riku_hisinputs.npy'
        load_fname = os.path.join('_data', fname)
        obs_hist_all = np.load(load_fname)[hdx,:,:]
        
        #plot historic runs
        for h,hist_name in enumerate(hist_name_list):
            print(hist_name)
            obs_hist=obs_hist_all[h].squeeze()

            fname = 'vae_riku_test_{:s}_k{:d}_e{:04d}.npy'.format(hist_name, kfold2use,epoch)
            load_fname = os.path.join('_output', fname)
            pred_hist = np.load(load_fname)

            pred_mean_hist=np.mean(pred_hist[:,:],axis=0)
            pred_2std_hist = 2 * np.std(pred_hist[:,:],axis=0)
        
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.cla()

            ax.errorbar(obs_hist,
                        pred_mean_hist,
                        yerr=pred_2std_hist,
                        fmt='.',
                        color='mediumblue',
                        elinewidth=0.8,
                        alpha=0.3,
                        capsize=0,
                        markersize=2)
            ax.text(0.1, 0.9, 'RSquare:' + str(round(r2_score(obs_hist, pred_mean_hist), 3)) + '\nG:' + str(round(Gfit(obs_hist, pred_mean_hist), 3)), color='k',ha='left', va='bottom',transform=plt.gca().transAxes)
            
            fig_title = \
                "Scatter of depths for {:s}".format(hist_name)
            ax.set_title(fig_title)
            ax.set_xlabel('observed flood depths(m)')
            ax.set_ylabel('predicted flood depths(m)')
            ax.grid(True, linestyle=':')           

            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            xylim = max(x1,y1) 
            ax.set_xlim([0.0, xylim +1])
            ax.set_ylim([0.0, xylim +1])
            ax.set_aspect('equal')

            fname = 'vae_depthsscatter_{:s}_k{:d}_e{:04d}.{:s}'.format(hist_name,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)
            fig.savefig(fname, dpi=dpi)
            plt.close(fig)

            # histogram of flood depths
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.cla()
            usebins=np.linspace(0.1, max(max(obs_hist),max(pred_mean_hist)) + 1, num=50)
            ax.hist(obs_hist, alpha=0.5,bins=usebins , label= 'obs', color='orange')
            ax.hist(pred_mean_hist, alpha=0.5, bins=usebins, label= 'pred', color='blue')
            ax.text(0.1, 0.9, 'RSquare:' + str(round(r2_score(obs_hist, pred_mean_hist), 3)) + '\nG:' + str(round(Gfit(obs_hist, pred_mean_hist), 3)),color='k',ha='left', va='bottom',transform=plt.gca().transAxes)

            fig_title = \
                "Histogram of depths >10cm for runno {:s}".format(hist_name)
            ax.set_title(fig_title)
            ax.set_ylabel('Count')
            ax.set_xlabel('Flood depths(m)')
            ax.legend(['observed', 'predicted'])

            fname = 'vae_depthshist_{:s}_k{:d}_e{:04d}.{:s}'.format(hist_name,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)
            fig.savefig(fname, dpi=dpi)
            plt.close(fig)

            #map plot of flood depths
            # colorbars
            cmap = plt.get_cmap('jet',5)          
            gridvalue = pred_mean_hist - obs_hist
            cig, px = plt.subplots(figsize=(3,1))
            norm = matplotlib.colors.Normalize(vmin=min(gridvalue), vmax=max(gridvalue))
            cbar = px.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=px, extend='neither', label = 'pred.mean - obs.(m)',orientation='horizontal')
            px.remove()
            cig.savefig('_plots/cbar.png',bbox_inches='tight')
            plt.close(cig)

            cig, px = plt.subplots(figsize=(3,1))
            norm = matplotlib.colors.Normalize(vmin=min(pred_mean_hist), vmax=max(pred_mean_hist))
            cbar = px.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=px, extend='neither', label = 'pred depth(m)',orientation='horizontal')
            px.remove()
            cig.savefig('_plots/depth.png',bbox_inches='tight')
            plt.close(cig)

            #contigency matrix
            threshold = 0.1
            O = obs_hist >= threshold
            NotO = obs_hist < threshold
            P = pred_mean_hist >= threshold
            NotP = pred_mean_hist < threshold

            print(h, hist_name)
            print('P:',sum(P))
            TP = np.logical_and(O, P)
            print('TP:',sum(TP))
            FP = np.logical_and(NotO, P)
            print('FP:',sum(FP))
            TN = np.logical_and(NotO, NotP)
            print('TN:',sum(TN))
            FN = np.logical_and(O, NotP)
            print('FN:',sum(FN))

            TPR = round(sum(TP)/(sum(TP)+sum(FN)),3)
            FPR = round(sum(FP)/(sum(FP)+sum(TN)),3)
            TNR = round(sum(TN)/(sum(FP)+sum(TN)),3)
            FNR = round(sum(FN)/(sum(FN)+sum(TP)),3)
            AC = round((sum(TP)+sum(TN))/(sum(FP)+sum(FN)+sum(TP)+sum(TN)),3)
            ER = round((sum(FP)+sum(FN))/(sum(FP)+sum(FN)+sum(TP)+sum(TN)),3)
            PR = round(sum(TP)/(sum(TP)+sum(FP)),3)

            # plots begin
            west, south, east, north = (140.85,37.76,141.09,38.33)
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 20))

            main_title = "Event {:s}".format(hist_name)
            fig.suptitle(main_title, fontsize=20)

            mx=ax[0,0]
            mx.cla()
            mx.set_xlim(west,east)
            mx.set_ylim(south,north)
            
            fig_title = "Flooded:TP(Blue)/FP(Red)" 
            mx.set_title(fig_title, fontsize=15)

            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(mx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(mx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            mx.scatter(fgmaxloc[TP,0], fgmaxloc[TP,1],s=0.5,c='b',alpha=0.6)
            mx.scatter(fgmaxloc[FP,0], fgmaxloc[FP,1],s=0.5,c='r',alpha=0.6)
            mx.text(140.95,38.10,s='TPR(TP/P):' + str(TPR), fontsize=12,color='white')
            mx.text(140.95,38.05,s='FPR(FP/N):' + str(FPR), fontsize=12,color='white')
            mx.text(140.95,38.00,s='Accuracy(True): ' + str(AC), fontsize=12,color='white')
            mx.text(140.95,37.95,s='Precision: ' + str(PR), fontsize=12,color='white')

            nx=ax[0,1]
            nx.cla()
            nx.set_xlim(west,east)
            nx.set_ylim(south,north)
            
            fig_title = "Not Flooded:TN(Green)/FN(Pink)"
            nx.set_title(fig_title, fontsize=15)

            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(nx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(nx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            nx.scatter(fgmaxloc[FN,0], fgmaxloc[FN,1],s=0.5,c='m',alpha=0.6)
            nx.scatter(fgmaxloc[TN,0], fgmaxloc[TN,1],s=0.5,c='g',alpha=0.6)
            nx.text(140.95,38.10,s='TNR(TN/N): ' + str(TNR), fontsize=12,color='white')
            nx.text(140.95,38.05,s='FNR(FN/P): ' + str(FNR), fontsize=12,color='white')
            nx.text(140.95,38.00,s='Error(False): ' + str(ER), fontsize=12,color='white')

            ox=ax[1,0]
            ox.cla()
            ox.set_xlim(west,east)
            ox.set_ylim(south,north)
            
            fig_title = "Predicted Depth(m)"
            ox.set_title(fig_title, fontsize=15)
            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')   

            ox.scatter(fgmaxloc[P,0], fgmaxloc[P,1], s=0.5, c=pred_mean_hist[P], cmap=cmap, alpha=0.6)
            ox.text(140.95,38.10,s='RSquare:' + str(round(r2_score(obs_hist, pred_mean_hist), 3)), fontsize=12,color='white')
            ox.text(140.95,38.05,s='G:' + str(round(Gfit(obs_hist, pred_mean_hist), 3)), fontsize=12,color='white')
            img = plt.imread('_plots/depth.png') 
            im = OffsetImage(img, zoom=.45)
            ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', box_alignment=(1.1,-0.1))
            ox.add_artist(ab)

            ox=ax[1,1]
            ox.cla()
            ox.set_xlim(west,east)
            ox.set_ylim(south,north)
            
            fig_title = "Error in Depth(m)"
            ox.set_title(fig_title, fontsize=15)
            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')                
            ox.scatter(fgmaxloc[P,0], fgmaxloc[P,1], s=0.5, c=gridvalue[P], cmap=cmap, alpha=0.6)
            img = plt.imread('_plots/cbar.png') 
            im = OffsetImage(img, zoom=.45)
            ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', box_alignment=(1.1,-0.1))
            ox.add_artist(ab)
        
            fname = 'vae_depthsmap_{:s}_k{:d}_e{:04d}.{:s}'.format(hist_name,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)

            # fig.tight_layout()
            fig.savefig(fname, dpi=dpi)
            plt.close(fig)
            
            ####Tohoku Comparison
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 20))
            ax.cla()
            ax.set_xlim(140.8434805670000003,141.1388615520000087)
            ax.set_ylim(37.7368119799999988,38.3404968680000025)

            fig_title = "Event {:s} vs 2011 TTJS Flood Extent".format(hist_name)
            ax.set_title(fig_title, fontsize=15)
            cx.add_basemap(ax, source='_data/fgmax001_Tohoku_base.tif', crs='EPSG:4326')   
            ax.scatter(fgmaxloc[P,0], fgmaxloc[P,1], s=1, c=pred_mean_hist[P],cmap=cmap, alpha=.25,
                       marker=',', edgecolors='none') 
            img = plt.imread('_plots/depth.png') 
            im = OffsetImage(img, zoom=.75)
            ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', box_alignment=(1.1,-0.1))
            ax.add_artist(ab)
            fname = 'vae_Tohoku_map_{:s}_k{:d}_e{:04d}.{:s}'.format(hist_name,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)

            fig.savefig(fname, dpi=dpi)
            plt.close(fig)    

        #plot test runs for a set of 5 events like historic runs
        for runno in test_runno[-5:]:
            pred_hist = pred_test[:,test_runno == runno,0,:].squeeze()
            obs_hist = obs_test[test_runno == runno,0,:].squeeze()
            if vae2use == 1:
                pred_mean_hist=pred_hist
                pred_2std_hist = np.zeros(pred_mean_hist.shape)
            else:
                pred_mean_hist=np.mean(pred_hist,axis=0)
                pred_2std_hist = 2 * np.std(pred_hist,axis=0)


            #scatter plot
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.cla()
            ax.errorbar(obs_hist,
                        pred_mean_hist,
                        yerr=pred_2std_hist,
                        fmt='.',
                        color='mediumblue',
                        elinewidth=0.8,
                        alpha=0.3,
                        capsize=0,
                        markersize=2)
            ax.text(0.1, 0.9, 'RSquare:' + str(round(r2_score(obs_hist, pred_mean_hist), 3)) + '\nG:' + str(round(Gfit(obs_hist, pred_mean_hist), 3)), color='k',ha='left', va='bottom',transform=plt.gca().transAxes)
            
            fig_title = \
                "Scatter of depths for runno {:4d}".format(runno)
            ax.set_title(fig_title)
            ax.set_xlabel('observed flood depths(m)')
            ax.set_ylabel('predicted flood depths(m)')
            ax.grid(True, linestyle=':')           

            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            xylim = max(x1,y1) 
            ax.set_xlim([0.0, xylim +1])
            ax.set_ylim([0.0, xylim +1])
            ax.set_aspect('equal')

            fname = 'vae_depthsscatter_test{:4d}_k{:d}_e{:04d}.{:s}'.format(runno,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)
            fig.savefig(fname, dpi=dpi)
            plt.close(fig)

            # histogram of flood depths
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            ax.cla()
            usebins=np.linspace(0.1, max(max(obs_hist),max(pred_mean_hist)) + 1, num=50)
            ax.hist(obs_hist, alpha=0.5,bins=usebins , label= 'obs', color='orange')
            ax.hist(pred_mean_hist, alpha=0.5, bins=usebins, label= 'pred', color='blue')
            ax.text(0.1, 0.9, 'RSquare:' + str(round(r2_score(obs_hist, pred_mean_hist), 3)) + '\nG:' + str(round(Gfit(obs_hist, pred_mean_hist), 3)),color='k',ha='left', va='bottom',transform=plt.gca().transAxes)

            fig_title = \
                "Histogram of depths >10cm for runno {:4d}".format(runno)
            ax.set_title(fig_title)
            ax.set_ylabel('Count')
            ax.set_xlabel('Flood depths(m)')
            ax.legend(['observed', 'predicted'])

            fname = 'vae_depthshist_test{:4d}_k{:d}_e{:04d}.{:s}'.format(runno,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)
            fig.savefig(fname, dpi=dpi)
            plt.close(fig)

            #map plot of flood depths
            # colorbars
            cmap = plt.get_cmap('jet',5)          
            gridvalue = pred_mean_hist - obs_hist
            cig, px = plt.subplots(figsize=(3,1))
            norm = matplotlib.colors.Normalize(vmin=min(gridvalue), vmax=max(gridvalue))
            cbar = px.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=px, extend='neither', label = 'pred.mean - obs.(m)',orientation='horizontal')
            px.remove()
            cig.savefig('_plots/cbar.png',bbox_inches='tight')
            plt.close(cig)

            cig, px = plt.subplots(figsize=(3,1))
            norm = matplotlib.colors.Normalize(vmin=min(pred_mean_hist), vmax=max(pred_mean_hist))
            cbar = px.figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),ax=px, extend='neither', label = 'pred depth(m)',orientation='horizontal')
            px.remove()
            cig.savefig('_plots/depth.png',bbox_inches='tight')
            plt.close(cig)

            #contigency matrix
            threshold = 0.1
            O = obs_hist >= threshold
            NotO = obs_hist < threshold
            P = pred_mean_hist >= threshold
            NotP = pred_mean_hist < threshold
            
            print(runno)
            print('P:',sum(P))
            TP = np.logical_and(O, P)
            print('TP:',sum(TP))
            FP = np.logical_and(NotO, P)
            print('FP:',sum(FP))
            TN = np.logical_and(NotO, NotP)
            print('TN:',sum(TN))
            FN = np.logical_and(O, NotP)
            print('FN:',sum(FN))

            TPR = round(sum(TP)/(sum(TP)+sum(FN)),3)
            FPR = round(sum(FP)/(sum(FP)+sum(TN)),3)
            TNR = round(sum(TN)/(sum(FP)+sum(TN)),3)
            FNR = round(sum(FN)/(sum(FN)+sum(TP)),3)
            AC = round((sum(TP)+sum(TN))/(sum(FP)+sum(FN)+sum(TP)+sum(TN)),3)
            ER = round((sum(FP)+sum(FN))/(sum(FP)+sum(FN)+sum(TP)+sum(TN)),3)
            PR = round(sum(TP)/(sum(TP)+sum(FP)),3)

            # plots begin
            west, south, east, north = (140.85,37.76,141.09,38.33)
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 20))

            main_title = "Event {:4d}".format(runno)
            fig.suptitle(main_title, fontsize=20)

            mx=ax[0,0]
            mx.cla()
            mx.set_xlim(west,east)
            mx.set_ylim(south,north)
            
            fig_title = "Flooded:TP(Blue)/FP(Red)" 
            mx.set_title(fig_title, fontsize=15)

            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(mx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(mx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            mx.scatter(fgmaxloc[TP,0], fgmaxloc[TP,1],s=0.5,c='b',alpha=0.6)
            mx.scatter(fgmaxloc[FP,0], fgmaxloc[FP,1],s=0.5,c='r',alpha=0.6)
            mx.text(140.95,38.10,s='TPR(TP/P):' + str(TPR), fontsize=12,color='white')
            mx.text(140.95,38.05,s='FPR(FP/N):' + str(FPR), fontsize=12,color='white')
            mx.text(140.95,38.00,s='Accuracy(True): ' + str(AC), fontsize=12,color='white')
            mx.text(140.95,37.95,s='Precision: ' + str(PR), fontsize=12,color='white')

            nx=ax[0,1]
            nx.cla()
            nx.set_xlim(west,east)
            nx.set_ylim(south,north)
            
            fig_title = "Not Flooded:TN(Green)/FN(Pink)"
            nx.set_title(fig_title, fontsize=15)

            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(nx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(nx, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            nx.scatter(fgmaxloc[FN,0], fgmaxloc[FN,1],s=0.5,c='m',alpha=0.6)
            nx.scatter(fgmaxloc[TN,0], fgmaxloc[TN,1],s=0.5,c='g',alpha=0.6)
            nx.text(140.95,38.10,s='TNR(TN/N): ' + str(TNR), fontsize=12,color='white')
            nx.text(140.95,38.05,s='FNR(FN/P): ' + str(FNR), fontsize=12,color='white')
            nx.text(140.95,38.00,s='Error(False): ' + str(ER), fontsize=12,color='white')

            ox=ax[1,0]
            ox.cla()
            ox.set_xlim(west,east)
            ox.set_ylim(south,north)
            
            fig_title = "Predicted Depth(m)"
            ox.set_title(fig_title, fontsize=15)
            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')   

            ox.scatter(fgmaxloc[P,0], fgmaxloc[P,1], s=0.5, c=pred_mean_hist[P], cmap=cmap, alpha=0.6)
            ox.text(140.95,38.10,s='RSquare:' + str(round(r2_score(obs_hist, pred_mean_hist), 3)), fontsize=12,color='white')
            ox.text(140.95,38.05,s='G:' + str(round(Gfit(obs_hist, pred_mean_hist), 3)), fontsize=12,color='white')
            img = plt.imread('_plots/depth.png') 
            im = OffsetImage(img, zoom=.45)
            ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', box_alignment=(1.1,-0.1))
            ox.add_artist(ab)

            ox=ax[1,1]
            ox.cla()
            ox.set_xlim(west,east)
            ox.set_ylim(south,north)
            
            fig_title = "Error in Depth(m)"
            ox.set_title(fig_title, fontsize=15)
            if not os.path.exists('_data/fgmax001_base.tif'):
                img, ext = cx.bounds2raster(west,south,east,north,ll=True,source=cx.providers.Esri.WorldImagery,path='_data/fgmax001_base.tif')
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')
            else:
                cx.add_basemap(ox, source='_data/fgmax001_base.tif', crs='EPSG:4326')                
            ox.scatter(fgmaxloc[P,0], fgmaxloc[P,1], s=0.5, c=gridvalue[P], cmap=cmap, alpha=0.6)
            img = plt.imread('_plots/cbar.png') 
            im = OffsetImage(img, zoom=.45)
            ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', box_alignment=(1.1,-0.1))
            ox.add_artist(ab)
        
            fname = 'vae_depthsmap_{:s}_k{:d}_e{:04d}.{:s}'.format(hist_name,kfold2use,epoch,fmt)
            fname = os.path.join('_plots', fname)

            # fig.tight_layout()
            fig.savefig(fname, dpi=dpi)
            plt.close(fig) 