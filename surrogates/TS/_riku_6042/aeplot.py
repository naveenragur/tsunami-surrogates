import os, sys
import numpy as np
import os, sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from pickle import load

def Gfit(obs, pred): #a normalized least-squares
    obs = np.array(obs)
    pred = np.array(pred)
    Gvalue = 1 - (2*np.sum(obs*pred)/(np.sum(obs**2)+np.sum(pred**2)))
    return Gvalue

def peak_dev(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    peak_dev = (np.max(np.abs(obs))-np.max(np.abs(pred)))/np.max(np.abs(obs))
    return peak_dev

def peak_delay(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    peak_delay = float(np.argmax(np.abs(obs))-np.argmax(np.abs(pred)))*240/1024
    return peak_delay

def plot_obsvpred(pred_train, obs_train,
                  pred_test, obs_test,
                  scale=1.0):
    r"""
    Plot prediction vs. observation
np.ceil
    Parameters
    ----------

    pred_train : array-like
        ensemble prediction time-series for train set with shape 
        (nensemble, ndata_train, ngauges, npts)

    obs_train : array-like
        time-series observation for test data
        (ndata_train, ngauges, npts)

    pred_test : array-like
        ensemble prediction time-series for test set with shape 
        (nensemble, ndata_test, ngauges, npts)

    obs_test : array-like
        time-series observation for test data
        (ndata_test, ngauges, npts)

    Returns
    -------
    fig_list : list
        a list of tuples containing figure, axis, gauge number

    """

    nensemble, ndata_train, ngauges, npts = pred_train.shape

    t_unif = np.linspace(0.0, 4.0, npts) # number of hours to plot

    pred_train_etamax = pred_train.max(axis=-1)
    pred_test_etamax = pred_test.max(axis=-1)

    obs_train_etamax = obs_train.max(axis=-1)
    obs_test_etamax = obs_test.max(axis=-1)

    predmean_train = np.mean(pred_train_etamax, axis=0)
    predmean_test = np.mean(pred_test_etamax, axis=0)

    pred2std_train = 2.0 * np.std(pred_train_etamax, axis=0)
    pred2std_test = 2.0 * np.std(pred_test_etamax, axis=0)

    # make a plot for output gauge
    obs_train_gaugei = obs_train_etamax[:, -1]
    obs_test_gaugei = obs_test_etamax[:, -1]

    predmean_train_gaugei = predmean_train[:,-1]
    predmean_test_gaugei = predmean_test[:, -1]

    pred2std_train_gaugei = pred2std_train[:, -1]
    pred2std_test_gaugei = pred2std_test[:, -1]

    vmax = max(predmean_train_gaugei.max(),
                predmean_test_gaugei.max(),
                obs_train_gaugei.max(),
                obs_test_gaugei.max())

    fig, ax = plt.subplots(figsize=(scale * 4, scale * 4))

    if vmax > 5.0:
        vmax = 25.0
    else:
        vmax = 5.0

    ax.plot([0.0, 1.10 * vmax],
            [0.0, 1.10 * vmax],
            '-k',
            linewidth=0.5)

    # plot obs/pred in train set
    line0, = ax.plot(obs_train_gaugei,
                        predmean_train_gaugei,
                        'k.',
                        alpha=0.3,
                        markersize=2,
                        zorder=0)

    # plot obs/pred in test set
    line1, = ax.plot(obs_test_gaugei,
                        predmean_test_gaugei,
                        linewidth=0,
                        marker='D',
                        color='tab:orange',
                        markersize=1.5,
                        zorder=4)

    # plot errorbars for predictions in test set
    ax.errorbar(obs_test_gaugei,
                predmean_test_gaugei,
                yerr=pred2std_test_gaugei,
                fmt='.',
                color='tab:orange',
                elinewidth=0.8,
                alpha=0.3,
                capsize=0,
                markersize=2)

    ax.text(10, 3.5, '$MSE Train$:' + str(round(mean_squared_error(obs_train_gaugei, predmean_train_gaugei), 3)), fontsize=8)
    ax.text(10, 2.5, '$MSE Test$:' + str(round(mean_squared_error(obs_test_gaugei, predmean_test_gaugei), 3)), fontsize=8)
    ax.text(10, 1.5, '$Gfit Train$:' + str(round(Gfit(obs_train_gaugei, predmean_train_gaugei), 3)), fontsize=8)
    ax.text(10, 0.5, '$Gfit Test$:' + str(round(Gfit(obs_test_gaugei, predmean_test_gaugei), 3)), fontsize=8)

    legends = [[line0, line1],
                ['train'+'('+ str(round(r2_score(obs_train_gaugei, predmean_train_gaugei), 3))+')',
                'test'+'('+ str(round(r2_score(obs_test_gaugei, predmean_test_gaugei), 3))+')']]

    ax.set_xlabel('observed $\eta_{max}$ ($m$)')
    ax.set_ylabel('predicted $\eta_{max}$ ($m$)')
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal')

    ax.set_xlim([0.0, vmax])
    ax.set_ylim([0.0, vmax])

    if vmax > 10:
        ax.xaxis.set_ticks(np.arange(0.0, vmax + 1, 2))
        ax.yaxis.set_ticks(np.arange(0.0, vmax + 1, 2))
    else:
        ax.xaxis.set_ticks(np.arange(0.0, vmax + 1))
        ax.yaxis.set_ticks(np.arange(0.0, vmax + 1))

    return  fig, ax, legends 

def plot_pred_timeseries(pred2, obs, gauges=[5832,6042],
                         scale=1.2, dpi=300):
    r"""
    Plot time-series prediction

    Parameters
    ----------

    pred2 : array-like
        ensemble prediction time-series with shape
        (2, nensemble, ngauges, npts)

    obs : array-like
        time-series observation data
        (ngauges, npts)

    ndata_train : int
        number of samples in training data

    Returns
    -------

    fig : pyplot figure

    """

    pred2 = np.array(pred2)
    npts = pred2.shape[-1]
    ngauges = len(gauges)

    t_unif = np.linspace(0.0, 4.0, npts) * 60.0

    fig, axes = plt.subplots(ncols=1,
                             nrows=ngauges,
                             figsize=(scale * 9, scale * ngauges * 8 / 6),
                             sharex=True,
                             sharey=True)
    ax = axes
    
    pred = pred2[...]


    color0 = plt.rcParams['axes.prop_cycle'].by_key()['color'] + \
                plt.rcParams['axes.prop_cycle'].by_key()['color'] + \
                plt.rcParams['axes.prop_cycle'].by_key()['color']+ \
                plt.rcParams['axes.prop_cycle'].by_key()['color']+ \
                plt.rcParams['axes.prop_cycle'].by_key()['color']

    vmax = np.max(np.abs(obs[ :]).max() * 1.1)
    vmax = np.ceil(vmax / 0.5) * 0.5

    for i in range(ngauges):
        ax = axes[i]
        predmean_gaugei = np.mean(pred[:, :], axis=0)
        pred2std_gaugei = 2 * np.std(pred[:, :], axis=0)
        obs_gaugei = obs[i,:]
    
        # plot prediction with uncertainty bands
        if i == ngauges-1:
            line1 = ax.fill_between(t_unif,
                                    predmean_gaugei - pred2std_gaugei,
                                    predmean_gaugei + pred2std_gaugei,
                                    facecolor=color0[i],
                                    alpha=0.30,
                                    edgecolor='k',
                                    linewidth=1.0)

            line0, = ax.plot(t_unif,
                            predmean_gaugei,
                            color=color0[i])

            ax.text(8, min(obs_gaugei)- 0.5,'Gfit:' + str(round(Gfit(obs_gaugei, predmean_gaugei), 3))
        + '  ,Rel_peakdev:' + str(round(peak_dev(obs_gaugei, predmean_gaugei), 3)) 
        + '  ,Peak_delay:' + str(round(peak_delay(obs_gaugei, predmean_gaugei),3))+ ' mins',
         fontsize=8, verticalalignment='top')
            
        # plot observed
        line2, = ax.plot(t_unif,
                            obs_gaugei,
                            linestyle="--",
                            color='k',
                            linewidth=1.0)


        # add legend
        line0_legend = '{:3d} pred'.format(gauges[i])
        line2_legend = '{:3d} obs'.format(gauges[i])

        if  i == ngauges-1:
            ax.legend((line0, line1, line2),
                        (line0_legend,
                        r'$\hat{\mu} \pm 2 \hat{\sigma}$',
                        line2_legend),
                        bbox_to_anchor=(1, 1),
                        loc='upper left')
        else:
            ax.legend((line2,),
                        (line2_legend,),
                        bbox_to_anchor=(1, 1),
                        loc='upper left')

        # set title
        ax.set_xlabel("time (mins)")
        ax.xaxis.set_ticks(np.arange(0, 4 * 61.0, 30))
        ax.set_xlim([0.0, 4 * 60.0])

    for i in range(ngauges):
        ax.set_ylim([-vmax, vmax])
        ax.set_ylabel("$\eta$ (m)")

    return fig, axes


def plot_pred_hist_timeseries(pred2, obs, gauges=[5832,6042],
                         scale=1.2, dpi=300):
    r"""
    Plot time-series prediction

    Parameters
    ----------

    pred2 : array-like
        ensemble prediction time-series with shape
        (2, nensemble, ngauges, npts)

    obs : array-like
        time-series observation data
        (ngauges, npts)

    ndata_train : int
        number of samples in training data

    Returns
    -------

    fig : pyplot figure

    """

    pred2 = np.array(pred2)
    npts = pred2.shape[-1]
    ngauges = len(gauges)

    t_unif = np.linspace(0.0, 4.0, npts) * 60.0

    fig, axes = plt.subplots(ncols=1,
                             nrows=ngauges,
                             figsize=(scale * 9, scale * ngauges * 8 / 6),
                             sharex=True,
                             sharey=True)


    ax = axes
    
    pred = pred2
    obs_in = obs[:]


    color0 = plt.rcParams['axes.prop_cycle'].by_key()['color'] + \
                plt.rcParams['axes.prop_cycle'].by_key()['color'] + \
                plt.rcParams['axes.prop_cycle'].by_key()['color']+ \
                plt.rcParams['axes.prop_cycle'].by_key()['color']+ \
                plt.rcParams['axes.prop_cycle'].by_key()['color']


    vmax = np.max(np.abs(obs[ :]).max() * 1.2)
    vmax = np.ceil(vmax / 0.5) * 0.5

    for i in range(ngauges):
        ax = axes[i]   
        predmean_gaugei = np.mean(pred[:, :], axis=0)
        pred2std_gaugei = 2 * np.std(pred[:,:], axis=0)
        obs_gaugei = obs[i, :]
        # plot prediction with uncertainty bands
        if i == ngauges - 1:
            line1 = ax.fill_between(t_unif,
                                    predmean_gaugei - pred2std_gaugei,
                                    predmean_gaugei + pred2std_gaugei,
                                    facecolor=color0[i],
                                    alpha=0.30,
                                    edgecolor='k',
                                    linewidth=1.0)

            line0, = ax.plot(t_unif,
                            predmean_gaugei,
                            color=color0[i])

            # line3, = ax.plot(t_unif,
            #                 np.median(pred[:,:], axis=0),
            #                 color='k',
            #                 linewidth=0.5)

        # plot observed
        line2, = ax.plot(t_unif,
                            obs_gaugei,
                            linestyle="--",
                            color='k',
                            linewidth=1.0)


        # add legend
        line0_legend = '{:3d} pred'.format(gauges[i])
        line2_legend = '{:3d} obs'.format(gauges[i])

        
        if i == ngauges-1:
            ax.text(8, min(obs_gaugei)- 0.5,'Gfit:' + str(round(Gfit(obs_gaugei, predmean_gaugei), 3))
        + '  ,Rel_peakdev:' + str(round(peak_dev(obs_gaugei, predmean_gaugei), 3)) 
        + '  ,Peak_delay:' + str(round(peak_delay(obs_gaugei, predmean_gaugei),3))+ ' mins',
         fontsize=8, verticalalignment='top')
            
            ax.legend((line0, line1, line2),
                        (line0_legend,
                        r'$\hat{\mu} \pm 2 \hat{\sigma}$',
                        line2_legend),
                        bbox_to_anchor=(1, 1),
                        loc='upper left')
        else:
            ax.legend((line2,),
                        (line2_legend,),
                        bbox_to_anchor=(1, 1),
                        loc='upper left')


        # set title
        ax.set_xlabel("time (mins)")
        ax.xaxis.set_ticks(np.arange(0, 4 * 61.0, 30))
        ax.set_xlim([0.0, 4 * 60.0])

    for i in range(ngauges):
        ax.set_ylim([-vmax, vmax])
        ax.set_ylabel("$\eta$ (m)")

    return fig, axes


def plot_pred_ensemble(pred_run, obs_run, gaugei=1):
    r"""
    Plot individual time-series predictions in the ensemble


    """

    t_unif = np.linspace(0.0, 4.0 * 60.0, 1024) #data points to recreate time series

    vmax = np.max([pred_run.max(), obs_run.max()])
    vmax = np.ceil(vmax / 0.5) * 0.5

    color0 = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,3))
    s0 = 1.2
    fh = s0 * 8.2
    fw = s0 * 2.2
    fig = plt.figure(figsize=(fh, fw))
    w = 0.86
    h = 0.86 * fh / fw * 0.1798
    ax = fig.add_axes([0.09, 0.25, w, h])

    if pred_run.shape[-1] < 50:
        lw = 1.0
        alpha = 0.5
    else:
        lw = 0.4
        alpha = 0.4


    line1 = ax.plot(t_unif,
                    pred_run,
                    color=color0[gaugei],
                    alpha=alpha,
                    linewidth=lw)
    line1 = line1[0]
    line0, = ax.plot(t_unif, obs_run, 'k--')

    ax.set_xlabel('time (mins)')
    ax.set_ylim([-vmax, vmax])
    ax.set_ylabel('$\eta$ (m)')
    ax.legend((line0, line1), ('obs', ' pred')) #replace with relavent gauge
    ax.set_xlim([0, 300])

    return fig, ax


def load_obspred(model_name,kfold, epochs_list,
                 model_fname_fmt='{:s}_test_{:02d}_{:04d}.npy'):
    r"""
    Plot time-series prediction

    Parameters
    ----------

    model_name : string
        name of the model. will load results with the filename

        '{:s}_test_{:02d}_{:04d}.npy'.format(model_name, obs_win, epoch)

        under the _output directory

    kfold : which fold to load data from train and test

    epochs_list : list of ints
        list of ints choosing the training epoch (for each obs window)

    Returns
    -------

    pred_train : array-like
        prediction results for the training set, of the shape
        (len(obs_win_list), nensemble, ndata_train, ngauges, npts)

    obs_train : array-like
        observations in the training set, of the shape
        (ndata_train, ngauges, npts)

    pred_test : array-like
        prediction results for the training set, of the shape
        (len(obs_win_list), nensemble, ndata_test, ngauges, npts)

    obs_test : array-like
        observations in the test set, of the shape
        (ndata_test, ngauges, npts)

    train_runno : array-like
        Geoclaw run numbers of the data in the training set

    test_runno : array-like
        Geoclaw run numbers of the data in the test set

    """

    # load all observed data
    data_dir = '_data'

    fname = os.path.join(data_dir, 'riku.npy')
    obs = np.load(fname)

    # load shuffled indices
    fname = os.path.join(data_dir, 'riku_train_index_k{:d}.txt'.format(kfold))
    train_index = np.loadtxt(fname).astype(int)

    fname = os.path.join(data_dir, 'riku_test_index_k{:d}.txt'.format(kfold))
    test_index = np.loadtxt(fname).astype(int)

    fname = os.path.join(data_dir, 'riku_train_runno_k{:d}.txt'.format(kfold))
    train_runno = np.loadtxt(fname).astype(int)

    fname = os.path.join(data_dir, 'riku_test_runno_k{:d}.txt'.format(kfold))
    test_runno = np.loadtxt(fname).astype(int)

    obs_train = obs[train_index, :, :]
    obs_test = obs[test_index, :, :]

    ndata_train = len(train_index)

    # prediction
    pred_train = []
    pred_test = []
    # Prediction vs. Observation plot

    # load prediction
    epoch = epochs_list

    fname = '{:s}_test_k{:d}_{:04d}.npy'.format(model_name, kfold, epoch)
    load_fname = os.path.join('_output', fname)
    pred_obs_kfold = np.load(load_fname)

    pred_train = pred_obs_kfold[:, :len(train_index), :, :]
    pred_test = pred_obs_kfold[:, -len(test_index):, :, :]

    return pred_train, obs_train, pred_test, obs_test, train_runno, test_runno


