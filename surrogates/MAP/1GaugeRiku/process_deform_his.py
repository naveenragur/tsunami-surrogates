"""
To check if footprints fgmax are generated and interpolate them as a gauge dataset
"""

import os, glob, shutil
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from clawpack.visclaw import colormaps
from clawpack.visclaw import plottools, gridtools
from clawpack.geoclaw import fgmax_tools, kmltools


# Environment variable
try:
    CLAW = os.environ['CLAW']
    HOME = os.environ['HOME']
    tohoku_dir = os.environ['PTHA']
    tsunami_dir = os.environ['TSU']

except:
    raise Exception("*** Must first set environment variable")

hist_name_list = ['FUJI2011_42','NANKAI2022','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD',
        'SatakeMiniLowerSoft','SatakeMiniUpper','SatakeMiniUpperSoft','SatakeMiniLower']

#Reads fgmax files and arranges them in a new file together with the corresponding runno as data file
def check_fgmax(data_path = '/mnt/data/nragu/Tsunami/Tohoku/_results/_output_SLAB/SL_{:04d}/fgmax{}.txt',
                his_path =  '/mnt/data/nragu/Tsunami/Tohoku/_results/_output6hrs/{}/fgmax{}.txt',
                dtopo_path = '/mnt/data/nragu/Tsunami/Tohoku/_tsunami/SL_{:04d}.tt3',
                his_dtopo_path = '/mnt/data/nragu/Tsunami/Tohoku/gis/{}.tt3',
                info_cols=[0,1,3], #lat long B
                data_cols=[4],#4 WL- lat long level B wl speed timewl timespeed timearrival
                runs = 771, #first 476 events are used ( ie 0-475 event no)
                fgmax_regions = ['0003']): #first 476 events are used ( ie 0-475 event no)
    run_data_buffer = []
    fgmax = {'0001':np.zeros((675311,3),dtype=float),'0002': np.zeros((172916,3),dtype=float), '0003':np.zeros((39516,3),dtype=float)} #size of fgmax rows with zeros
    hisfgmax = {'0001':np.zeros((675311,3),dtype=float), '0002':np.zeros((172916,3),dtype=float), '0003':np.zeros((39516,3),dtype=float)}
    
    #create data path
    if not os.path.isdir('_data'):
        os.mkdir('_data')
    
    # Read fgmax data:
    fg = fgmax_tools.FGmaxGrid()
    fgmax_input_file_name = '/mnt/data/nragu/Tsunami/Tohoku/_results/_output_SLAB/SL_0000/fgmax_grids.data'

    for fgmax_no in fgmax_regions: #iterate regions
        print('fgmax_no:',fgmax_no)
        fg.read_fgmax_grids_data(fgno=int(fgmax_no), data_file=fgmax_input_file_name)
    
        for hist_name in hist_name_list:
                fname = his_path.format(hist_name, fgmax_no)
                if not os.path.exists(fname):
                    print('File {} does not exist'.format(fname))
                    run_data_buffer.append(fname)
                else:
                    data_fgm = np.loadtxt(fname)
                    fg.read_output(outdir='/mnt/data/nragu/Tsunami/Tohoku/_results/_output6hrs/{:s}/'.format(hist_name))
                    fg.interp_dz('/mnt/data/nragu/Tsunami/Tohoku/gis/dtopo_his/{:s}.tt3'.format(hist_name), dtopo_type=3) #dtopo_his
                    h = fg.h.T.flatten()
                    dZ = fg.dz.T.flatten()
                    # B = fg.B.T.flatten()
                    # X = fg.X.T.flatten()
                    # Y = fg.Y.T.flatten()     
                    # h0 = h# + dZ 
                    # h0 = h0.reshape(-1,1)  
                    dZ = dZ.reshape(-1,1)  
                    hisfgmax[fgmax_no][:,0:3] = data_fgm[:, info_cols]  
                    hisfgmax[fgmax_no]=np.append(hisfgmax[fgmax_no],dZ, axis = 1)
                    sys.stdout.write(
                    '\nhistorical event = {}, fgmax_regions = {} does exists and processing\n'.format(hist_name, fgmax_no))
    
        np.savetxt('_data/fgdz'+ fgmax_no +'_hisdata.csv', hisfgmax[fgmax_no], delimiter=',', fmt='%10.5f')  

    if len(run_data_buffer) > 0:
        print('missing runs: ',run_data_buffer)

# filters the fgmax data to land and only flooding regions
def select_fgmax(his_path_fgmax ='_data/fgmax{}_hisflooded.csv',
                 his_path_dz = '_data/fgdz{}_hisdata.csv',
                 fgmax_regions = ['0003']): #first 476 events are used ( ie 0-475 event no)):
    
    for fgmax_no in fgmax_regions:    
        
        #load data synthetic and historical
        hisfgmax=np.loadtxt(his_path_fgmax.format(fgmax_no), delimiter=',')
        hisfgdz=np.loadtxt(his_path_dz.format(fgmax_no), delimiter=',')
        
        print(hisfgmax.shape)
        print(hisfgdz.shape)

        #filter only rows of hisfgdz with same lat-column 0 and long-column 1 as in hisfgmax
        # Create keys for hisfgdz and hisfgmax
        keys_hisfgdz = tuple(map(tuple, hisfgdz[:, 0:2]))
        keys_hisfgmax = tuple(map(tuple, hisfgmax[:, 0:2]))

        # Find matching keys using set intersection
        matching_keys = set(keys_hisfgdz).intersection(keys_hisfgmax)

        # Get the indices of matching entries in hisfgdz
        matching_indices = [i for i, key in enumerate(keys_hisfgdz) if key in matching_keys]
        hisfgdz = hisfgdz[matching_indices]
        
        #reorder indices as per keys_hisfgmax
        keys_hisfgdz = tuple(map(tuple, hisfgdz[:, 0:2]))
        reorder_indices = [keys_hisfgdz.index(key) for key in keys_hisfgmax]
        hisfgdz = hisfgdz[reorder_indices]
        print(hisfgdz.shape)
        np.savetxt('_data/fgdz{}_hisflooded.csv'.format(fgmax_no), hisfgdz, delimiter=',', fmt='%10.5f')  

if __name__ == "__main__":
    check_fgmax()
    select_fgmax()
