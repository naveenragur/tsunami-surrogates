"""
To process folder names and gauge data from synthetic runs to usable format for ML

Interpolate to a uniform grid of a time-window

Notes
-----
After running this script: to access interpolated train and test sets

    data = np.load('_data/riku.npy')
    train_index = np.load('_data/riku_train_index.txt')
    test_index = np.load('_data/riku_test_index.txt')

    data_train = data[train_index, ...]
    data_test = data[test_index, ...]

"""

import os, glob, shutil
import sys
from scipy.signal import find_peaks
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pickle import dump
# import seaborn as sns

# Environment variable
try:
    CLAW = os.environ['CLAW']
    HOME = os.environ['HOME']
    tohoku_dir = os.environ['PTHA']
    tsunami_dir = os.environ['TSU']
    
except:
    raise Exception("*** Must first set environment variable")

#functions to preprocess to ML input requirements
def interp_gcdata(npts=1024, #64 observations per hour ie ~ 1 per minute
                  nruns=771, #number of runs
                  gauge_path_format= \
                          '/mnt/data/nragu/Tsunami/Tohoku/_results/_output_SLAB/SL_{:04d}/gauge{:05d}.txt',
                  filter_path= \
                          '/mnt/data/nragu/Tsunami/Tohoku/gis/slab_dtopo_list2run4footprint.csv',  #slab_dtopo_list2run4footprint
                  gauges=[5010,5015,5020,5675],
                  gauge_data_cols=[1, 5],
                  skiprows=3,
                  thresh_vals=[0.1, 0.5], #offshore and nearshore threshold
                  win_length=4*3600.0,  #filter with min time window
                  data_path='_data',
                  dataset_name='riku',
                  make_plots=False,
                  make_stats=False,
                  use_Agg=False):
    '''
    Process GeoClaw gauge output and output time-series interpolated on a
    uniform grid with npts grid pts.


    Parameters
    ----------

    npts : int, default 1024
        no of pts on the uniform time-grid

    skiprows : int, default 2 but in processed file 0
        number of rows to skip reading GeoClaw gauge output

    nruns : int, default 240 scenarios
        total number of geoclaw runs

    gauge_path_format : string, default  'run_{:06d}/_output/gauge_{:05d}.txt'
        specify sub-directory format of geoclaw output, for example
            'run_{:06d}/_output/gauge_{:05d}.txt'
        the field values will contain run and gauge numbers

    gauges : list of ints, default [6042, 5901, 5845, 5832] for riku
        specify the gauge numbers

    gauge_data_cols : list of ints, default [1, 5]
        designate which columns to use for time and surface elevation
        in the GeoClaw gauge ouput file

    skiprows : int, default  2
        number of rows to skip when reading in the GeoClaw gauge output file

    thresh_vals : list of floats, [0.1, 0.5]
        designate threshold values to impose on the gauge data:
        excludes all runs with:
        abs(eta) < thresh_vals[0] in the first gauge(offshore)
        abs(eta) < thresh_vals[1] in the last gauge(nearshore)

    win_length : float, default 4*3600.0,
        length of the time window in seconds

    data_path : str, default '_data'
        output path to save interpolation results and other information

    dataset_name : str, default 'riku'
        name the dataset

    make_plots : bool, default False
        set to True to generate individual plots for each of the runs

    use_Agg : bool, default False
        set to True to use 'Agg' backend for matplotlib, relevant only when
        kwarg make_plots is set to True

    '''

    #create data path
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    ngauges = len(gauges)

    data_all = np.zeros((nruns, ngauges, npts))
    i_valid = np.zeros(nruns, dtype=np.bool_)
    t_win_all = np.zeros((nruns, 2))
    run_stat=np.zeros((nruns, ngauges,6)) #[peak value, peak time, no of peaks, Name, Mag,dz]

    if use_Agg:
        import matplotlib
        matplotlib.use('Agg')

    if make_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

    filter_list=np.loadtxt(filter_path, skiprows=1, delimiter=',',usecols=(1,2,9))
  
    for i, run_no in enumerate(range(nruns)):
        
        run_data_buffer = []
        dz_buffer = []
        
        # collect gauge data from run
        for k, gauge_no in enumerate(gauges):
            gauge_fname = gauge_path_format.format(run_no, gauge_no)
            data = np.loadtxt(gauge_fname, skiprows=skiprows)#, delimiter=',')
            dz = data[1,5] # the deformation at gauge

            sys.stdout.write(
                '\rrun_no = {:6d}, data npts = {:6d}, unif npts = {:d}, gaugeno = {:d}, deformation = {:4.3f}'.format(
                    run_no, data.shape[0], npts,gauge_no, dz))

            # extract relevant columns
            data[:,5] = data[:,5] - dz #correction of local deformation
            run_data_buffer.append(data[1:, gauge_data_cols]) 
            dz_buffer.append(dz)


            if make_plots:
                if not os.path.isdir("_gaugeplots"):
                    os.mkdir("_gaugeplots")
                ax.cla()
                ax.plot(data[1:, gauge_data_cols[0]], (data[1:, gauge_data_cols[1]]))
                ax.axhline(y = dz, color = 'r', linestyle = '-')
                #add deformation value in plot
                ax.text(0.5, 0.5, 'Deformation = {:4.3f}'.format(dz), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                # ax.plot(t_unif, eta_unif)
                fig_title = \
                    "run_no {:06d}, gauge {:05d}".format(run_no, gauge_no)
                ax.set_title(fig_title)

                fig_fname = \
                    '_gaugeplots/run_{:06d}_gauge{:05d}.png'.format(run_no, gauge_no)
                fig.savefig(fig_fname, dpi=200)

        valid_data, t_win_lim = _get_window(run_data_buffer,
                                            thresh_vals,
                                            win_length)
       
        if filter_list[i,2] and valid_data:
            i_valid[i] = 1
        else:
            i_valid[i] = 0

        if valid_data :
            t0 = t_win_lim[0]
            t1 = t_win_lim[1]
            t_unif = np.linspace(t0, t1 + 00 * 60.0, npts)

            for k in range(ngauges):
                raw = run_data_buffer[k]
                dz = dz_buffer[k]
                eta_unif = np.interp(t_unif, raw[:, 0], raw[:, 1])
                data_all[run_no, k, :] = eta_unif
    
                # peak = np.amax(eta_unif)
                # peakelement= (int(np.where(eta_unif == peak)[0])*win_length)/(npts*3600)

                peak = np.amax(np.abs(eta_unif))
                peakelement = (np.argmax(np.abs(eta_unif))*win_length)/(npts*3600)
                peaks = len(find_peaks(x=np.abs(eta_unif),height = peak*.5))
                stats = [peak,peakelement,peaks,int(filter_list[i,0]),filter_list[i,1],dz]
                run_stat[run_no, k, :] = stats #peak value, peak time,no of peaks, even name, Mag, dz

            t_win_all[run_no, 0] = t0 
            t_win_all[run_no, 1] = t1
            sys.stdout.write('   --- valid --- ')
        else:
            pass

    data_all = data_all[i_valid, :, :]
    t_win_all = t_win_all[i_valid, :]
    run_stat = run_stat[i_valid, :, :] 
    
    if make_stats:
        import matplotlib.pyplot as plt
        fig, sx = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        if not os.path.isdir("_gaugeplots"):
            os.mkdir("_gaugeplots")

        for k, gauge_no in enumerate(gauges):
            ax=sx
            # frequency plot
            data=run_stat[:,k,0]
            ax.cla()
            ax.hist(data, weights=np.ones_like(data) / len(data))
            # ax.plot(t_unif, eta_unif)
            fig_title = \
                "Max Amplitude at gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Max Amplitude ')
            ax.set_ylabel('Frequency')
            ax.set_xlim(0, 10)

            fig_fname = \
                '_gaugeplots/freq_maxwl_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)

            #Time of max amplitude
            data=run_stat[:,k,1]
            ax.cla()
            ax.hist(data, weights=np.ones_like(data) / len(data))
            # ax.plot(t_unif, eta_unif)
            fig_title = \
                "Max Amplitude Time at gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Time of Max Amplitude ')
            ax.set_ylabel('Frequency')
            ax.set_xlim(0, 4)

            fig_fname = \
                '_gaugeplots/freq_maxT_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)
           
            # box plots

            data = pd.DataFrame({'Mag': run_stat[:,k,-2],
            'MaxAmp': run_stat[:,k,0],'MaxAmpTime': run_stat[:,k,1]})
            
            data75=data[data['Mag']==7.5]
            data80=data[data['Mag']==8]
            data85=data[(data['Mag']>=8.5) & (data['Mag'] < 8.75)]
            data90=data[data['Mag'] > 8.75]

            #mag plot - wl
            ax.cla()
            ax.boxplot([data75['MaxAmp'],data80['MaxAmp'],data85['MaxAmp'],data90['MaxAmp']])
            ax.set_xticklabels( ['7.5', '8', '8.5', '9'])

            fig_title = \
                "MaxAmp at diff mag for gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Magnitude ')
            ax.set_ylabel('Max Amplitude')


            fig_fname = \
                '_gaugeplots/scatter_mag_maxwl_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)

            #mag plot - time of peak
            ax.cla()
            ax.boxplot([data75['MaxAmpTime'],data80['MaxAmpTime'],data85['MaxAmpTime'],data90['MaxAmpTime']])
            ax.set_xticklabels( ['7.5', '8', '8.5', '9'])
            fig_title = \
                "Time of Max Amplitude at diff mag for gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Magnitude ')
            ax.set_ylabel('Time of Max Amplitude')
  
            fig_fname = \
                '_gaugeplots/scatter_mag_maxtime_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)

            # save stats #peak value, peak time,no of peaks, even name, Mag, dz
            output_fname = os.path.join(data_path,
                                        '{:05d}_stats.txt'.format(gauge_no))
            np.savetxt(output_fname, run_stat[:,k,:], fmt='%1.3f')



    # save eta
    output_fname = os.path.join(data_path,
                                '{:s}gauge.npy'.format(dataset_name))
    np.save(output_fname, data_all)

    # save uniform time grid
    output_fname = os.path.join(data_path,
                                '{:s}_t.npy'.format(dataset_name))
    np.save(output_fname, t_win_all)

    # save picked run numbers

    runnos = np.arange(nruns)[i_valid]
    output_fname = os.path.join(data_path,
                                '{:s}_runno.txt'.format(dataset_name))
    np.savetxt(output_fname, runnos, fmt='%d')


def _get_window(run_data, thresh_vals, win_length):
    r"""
    Check if data satisfies the requirements: threshold eta at near and offshore

    Parameters
    ----------
    run_data :
        list containing unstructure time reading from the three gauges

    thresh_vals :
        array of size 2 with thresholds for 5832,6042

    win_length :
        length of the prediction window (in seconds)

    Returns
    -------
    valid_data : bool
        True if the run data can be thresholded / windowed properly

    t_win_lim : array
        2-array with beginning and ending time point

    """

    ngauges = len(run_data)
    flag = np.zeros(2, dtype=np.bool_)

    t_win_lim = np.zeros(2)

    # apply threshold to 5832(offshore gauge)
    gaugei_data = run_data[0] #1#index of representative offshore gauge

    t = gaugei_data[:, 0]
    eta = gaugei_data[:, 1]
    t_init = t[0]
    t_final = t[-1]

    i_thresh = (np.abs(eta) >= thresh_vals[0])

    if (np.sum(i_thresh) > 0):
        t_win_init = np.min(t[i_thresh])
        t_win_final = t_win_init + win_length
        t_win_lim[0] = t_win_init
        t_win_lim[1] = t_win_final

        if t_win_final <= t_final:
            flag[0] = True

    # apply threshold to 6042(nearshore)
    gaugei_data = run_data[-1] #-2#index of representative nearshore gauge

    t = gaugei_data[:, 0]
    eta = gaugei_data[:, 1]

    i_thresh = (np.abs(eta) >= thresh_vals[1])

    if (np.sum(i_thresh) > 0):
        flag[1] = True

    # are both thresholds satisfied?
    valid_data = np.all(flag)
    return valid_data, t_win_lim

def shuffle_dataset(dataset_name='riku', data_path='_data',seed=99999, folds = 5):
  
    """
    Shuffle interpolated dataset. Run after interp_gcdata()

    Parameters
    ----------

    dataset_name : str, default 'riku'
        set dataset_name, the function requires the file
            '{:s}/{:s}_runno.npy'.format(data_path, dataset_name)

    data_path : str, default '_data'
        output path to save interpolation results and other information

    seed : int, default 12345
        Random seed supplied to np.random.shuffle()

    training_set_ratio : float, default 0.8
    the ratio of total runs to be set as the training set

    Notes
    -----

    Shuffled GeoClaw run numbers or indices are saved in
       '{:s}/{:s}_train_runno.txt'.format(data_path, dataset_name)
       '{:s}/{:s}_train_index.txt'.format(data_path, dataset_name)
       '{:s}/{:s}_test_runno.txt'.format(data_path, dataset_name)
       '{:s}/{:s}_test_index.txt'.format(data_path, dataset_name)

    """

    np.random.seed(seed)

    fname = '{:s}_runno.txt'.format(dataset_name)
    full_fname = os.path.join(data_path, fname)
    gc_runno = np.loadtxt(full_fname)
    ndata = len(gc_runno)
    shuffled_indices = np.arange(ndata)
    np.random.shuffle(shuffled_indices)
    shuffled_gc_runno = gc_runno[shuffled_indices]
    fold_size = ndata // folds
   
    for k in range(folds):
        print("shuffling fold {:d}".format(k))    

        fold_start = k * fold_size #0,50,100,150,200
        fold_end = (k + 1) * fold_size #50,100,150,200,250

        train_index = np.concatenate((shuffled_indices[:fold_start], shuffled_indices[fold_end:]))
        test_index = shuffled_indices[fold_start:fold_end]

        train_runno = gc_runno[train_index]
        test_runno = gc_runno[test_index]

        # filenames to save
        output_list = [('{:s}_train_runno_k{:d}.txt'.format(dataset_name,k), #till train indices
                        train_runno),
                    ('{:s}_train_index_k{:d}.txt'.format(dataset_name,k),
                        train_index),
                    ('{:s}_test_runno_k{:d}.txt'.format(dataset_name,k), #from train indices
                        test_runno),
                    ('{:s}_test_index_k{:d}.txt'.format(dataset_name,k),
                        test_index)]
        for fname, array in output_list:
            full_fname = os.path.join(data_path, fname)
            np.savetxt(full_fname, array, fmt='%d')
    print("done shuffling!")    
    np.savetxt('_data/{:s}_runno_sh.txt'.format(dataset_name),shuffled_gc_runno,fmt='%d')
                  

if __name__ == "__main__":
    # threshold, interpolate on window, save data
    interp_gcdata(make_plots=True,make_stats=True, use_Agg=True)
    
    # shuffle run numbers, separate training vs test sets for k fold cross validation, store data
    shuffle_dataset(data_path='_data', dataset_name='riku',folds = 5)


