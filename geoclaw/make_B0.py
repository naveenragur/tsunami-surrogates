import os
import numpy as np
from clawpack.geoclaw import fgmax_tools

#start processing B0 npy files from fgmax output
outdir = '_output/B0'
os.makedirs('_plots', exist_ok = True)

os.system('make B0data') #check if making_B0=True in setrun is there
os.system('make B0run')
os.system('make B0plots')

# Read fgmax data:
fg = fgmax_tools.FGmaxGrid()
fgmax_input_file_name = outdir + '/fgmax_grids.data'
print('fgmax input file: \n  %s' % fgmax_input_file_name)
res = ['dx150', 'dx050', 'dx050'] #if using multiple resolutions

for fgno in [1, 3]: #depending on region setup for fgmax for now only 1 on sendai, and 3 if all innundation done
    fg.read_fgmax_grids_data(fgno=fgno, data_file=fgmax_input_file_name)
    fg.read_output(outdir=outdir)
    B0 = np.array(fg.B)
    name = os.path.join(outdir,('fgmax000%s_' + res[fgno]) % fgno)
    print('Creating %s npy files' % name)
    np.save(name + '_X.npy', fg.X)
    np.save(name + '_Y.npy', fg.Y)
    np.save(name + '_B0.npy', B0)
