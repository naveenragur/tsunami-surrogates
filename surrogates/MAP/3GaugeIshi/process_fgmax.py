"""
To check if footprints fgmax are generated and interpolate them as a gauge dataset
"""

import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


#Combines the two datasets - gauge and footprint
def add_footprint2gauge(data_path ='_data/fgmax{}_flooded.csv',
                        numpy_path ='_data/rikugauge.npy',
                        runs_path= '_data/riku_runno.txt',
                        fgmax_regions = ['0002']): 

    for fgmax_no in fgmax_regions:    
        
        filtereve=np.loadtxt(runs_path,dtype=int)
        print('no of events', len(filtereve))

        fgmax=np.loadtxt(data_path.format(fgmax_no), delimiter=',')
        fgmax_depth=fgmax[:,3:] #only depths not lat long
        
        fgmax_depth=fgmax_depth[:,filtereve]
        print('size after all filtering of events:',fgmax_depth.shape)
       
        restacked = np.dstack(fgmax_depth[:]).transpose(1,0,2)
        print('shape of inundation data:',restacked.shape)

        gauge=np.load(numpy_path)
        gaugein = gauge[:,:-1,:].reshape(gauge.shape[0],3,gauge.shape[2])
        print('shape of gauge data:',gaugein.shape)

        np.save('_data/riku_floodinputs.npy',restacked)
        np.save('_data/riku_gaugeinputs.npy',gaugein)

def add_hisfootprint2hisgauge(data_path ='_data/fgmax{}_hisflooded.csv',
                            fgmax_regions = ['0002']): 

    for fgmax_no in fgmax_regions:    
        
        fgmax=np.loadtxt(data_path.format(fgmax_no), delimiter=',')
        print('shape of historical data:',fgmax.shape)
        fgmax_depth=fgmax[:,3:]
        restacked = np.dstack(fgmax_depth[:]).transpose(1,0,2)
        print(restacked.shape)
            
        np.save('_data/riku_hisinputs.npy',restacked)

if __name__ == "__main__":
    add_footprint2gauge()
    add_hisfootprint2hisgauge()
