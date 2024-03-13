# Process fgmax grid results and plot --- Sendai Bay
# To process fgmax results after doing a run.

from genericpath import exists
import matplotlib

matplotlib.use('Agg')
import pandas as pd
from pylab import *
import os, sys, glob
import matplotlib as mpl
from matplotlib import colors
from clawpack.visclaw import colormaps
from clawpack.visclaw import plottools, gridtools
from clawpack.geoclaw import fgmax_tools, kmltools
import warnings

warnings.filterwarnings("ignore")

def process_fg_max(outdir='_output', plotdir='_plots', dtopo=None, save_figs=True, testdata = False, dtopodir='_tsunami'):

    root_dir = os.environ['PTHA']  # should point to main repo directory
    print('using root_dir = ', root_dir)

    outdir = './' + outdir + '/' + dtopo.split('.t')[0]
    plotdir = './' + plotdir + '/' + dtopo.split('.t')[0]

    #-------------------------------
    # test data only:
    if testdata:
        outdir = './_output/tohoku_dtopo_2011'
        plotdir = './_plots/tohoku_dtopo_2011'
    #-------------------------------

    file = open(os.path.join(outdir,'dtopo.data'), 'r')
    test_dtopo = file.readlines()[6][0]
    file.close()

    if test_dtopo == '0':
        dtopo_path = None
        print('Run used no dtopofile')

    if test_dtopo == '1':
        file = open(os.path.join(outdir, 'dtopo.data'), 'r')
        dtopo_lines = file.readlines()
        file.close()
        dtopo_path = dtopo_lines[9][1:-2]
        dtopo_type = int(dtopo_lines[10][0])
        if exists(dtopo_path):
            print('Run used dtopofile = %s of type %i' %(dtopo_path, dtopo_type))
        else:
            dtopo_path = os.path.join(dtopodir,os.path.basename(dtopo_path))
            print('Run used dtopofile = %s of type %i' %(dtopo_path, dtopo_type))

    print('Will read fgmax results from outdir = \n  ', outdir)

    fgmax_plotdir = plotdir + '/fgmax_plots'
    gis_dir = plotdir + '/fgmax_gis'

    print('Will send plots to fgmax_plotdir = \n  ', fgmax_plotdir)
    print('Will send gis data to gis_dir = \n  ', gis_dir)
    os.system('mkdir -p %s' % fgmax_plotdir);
    os.system('mkdir -p %s' % gis_dir);

    def savefigp(fname):
        if save_figs:
            fullname = '%s/%s' % (fgmax_plotdir, fname)
            savefig(fullname)
            print('Created ', fullname)
        else:
            print('save_figs = False')

    # #Read in and process the fgmax results from the latest run

    print('outdir = ',outdir)
    t_files = glob.glob(outdir + '/fort.t0*')
    times = []
    for f in t_files:
        lines = open(f,'r').readlines()
        for line in lines:
            if 'time' in line:
                t = float(line.split()[0])
        times.append(t)
    times.sort()
    print('Output times found: ',times)
    if len(times) > 0:
        t_hours = times[-1] / 3600.
        print('\nfgmax results are presumably from final time: %.1f seconds = %.2f hours'\
               % (times[-1], t_hours))
    else:
        t_hours = nan

    # Read fgmax data:
    fg = fgmax_tools.FGmaxGrid()
    fgmax_input_file_name = os.path.join(outdir, 'fgmax_grids.data')
    print('fgmax input file: \n  %s' % fgmax_input_file_name)

    for fgno in range(1,4):
        print('processing fgmax grid no:', fgno)
        fg.read_fgmax_grids_data(fgno=fgno, data_file=fgmax_input_file_name)

        fg.read_output(outdir=outdir)

        coarse_run = (fg.X[1,0]-fg.X[0,0] > .001) #based on resolution

        print('Number of fgmax points: ', fg.h.count())

        # Determine B0, original topo before subsidence/uplift

        # old way, from dtopo:

        if dtopo_path is not None:
            # compute subsidence/uplift at each fgmax point:
            print('Interpolating dtopo')
            fg.interp_dz(dtopo_path, dtopo_type=dtopo_type)  # sets fg.dz based on dtopo
                                                             # dtopo_type either 1 or 3
                                                             # is set at topo of pgm by
                                                             # Loyce
        else:
            print('No uplift or subsidence in fgmax region')
            fg.dz = zeros(fg.B.shape)

        B0_from_dtopo = fg.B - fg.dz

        #Read in saved B0 .npy file from no tsunami run called B0:
        if (coarse_run == True):
            print('This was a coarse grid run')
            run = 'dx150_fg' + str(fgno)
            res = 0.00135

            Nrep = 1
        else:
            print('This was a fine grid run')
            run = 'dx050_fg' + str(fgno)
            res = 0.00045
            Nrep = 1     #No need for replicating, see below

        # B0 = load(root_dir + '/_output/B0/fgmax0001_' + run + '_B0.npy')
        # X0 = load(root_dir + '/_output/B0/fgmax0001_' + run + '_X.npy')
        # Y0 = load(root_dir + '/_output/B0/fgmax0001_' + run + '_Y.npy')

        # assert abs(fg.X-X0).max() == 0., "X0 should match fg.X"
        # assert abs(fg.Y-Y0).max() == 0., "Y0 should match fg.Y"

        # dB0 = abs(B0_from_dtopo - B0).max()
        # print("max diff in B0 from dtopo version = %g" % dB0)

        # Use the saved version:
        # fg.B0 = B0
        fg.B0 = B0_from_dtopo
        zmin = -60.
        zmax = 40.
        land_cmap = colormaps.make_colormap({0.0:[0.1,0.4,0.0],
                                             0.25:[0.0,1.0,0.0],
                                              0.5:[0.8,1.0,0.5],
                                              1.0:[0.8,0.5,0.2]})

        sea_cmap = colormaps.make_colormap({ 0.0:[0,0,1], 1.:[.8,.8,1]})

        cmap, norm = colormaps.add_colormaps((land_cmap, sea_cmap),
                                             data_limits=(zmin,zmax),
                                             data_break=0.)

        #plot for cosesmic displacement
        def plotZ(Z, show_cb=True):
            print('+++ fg.Y.shape, Z.shape: ',fg.Y.shape, Z.shape)
            print('+++ fg.Y[:2,:2] ',fg.Y[:2,:2])
            pc = plottools.pcolorcells(fg.X, fg.Y, Z, cmap=cmap, norm=norm)
            if show_cb:
                cb = colorbar(pc,shrink=0.5)
                cb.set_label('meters')
            #axis([-122.76,-122.525,47.95,48.2])
            gca().set_aspect(1./cos(38*pi/180.))
            ticklabel_format(useOffset=False)
            xticks(rotation=20);

        figure(figsize=(12,12))
        subplot(121)
        plotZ(fg.B, show_cb=False)
        title('GeoClaw B (post-seismic)');

        subplot(122)
        plotZ(fg.B0, show_cb=False)
        title('B0 (pre-seismic)');
        tight_layout()

        savefigp('geoclaw_topo_' + run + '.png')

        onshore = (fg.B0 > 0)
        offshore = logical_not(onshore)

        fg.h_onshore = ma.masked_where(offshore, fg.h)

        ###  Loyce:  eta = B0 + h, not eta = B + h for this project which is after the cosesmic deformation topo level
        #fg.eta_offshore = ma.masked_where(onshore, fg.B + fg.h)
        fg.eta_offshore = ma.masked_where(onshore, fg.B0 + fg.h)


        #bounds_depth = array([1e-6,0.25,0.5,0.75,1,1.25,1.5])
        bounds_depth = array([1e-6,0.5,1.0,1.5,2,2.5,3.0])


        cmap_depth = colors.ListedColormap([[.7,.7,1],[.5,.5,1],[0,0,1],[1,.7,.7],\
                                            [1,.4,.4], [1,0,0]])

        # Set color for value exceeding top of range to purple:
        cmap_depth.set_over(color=[1,0,1])

        # Set color for land points without inundation to light green:
        cmap_depth.set_under(color=[.7,1,.7])

        norm_depth = colors.BoundaryNorm(bounds_depth, cmap_depth.N)


        figure(figsize=(10,10))
        pc = plottools.pcolorcells(fg.X, fg.Y, fg.h_onshore, cmap=cmap_depth, norm=norm_depth)
        cb = colorbar(pc, extend='max', shrink=0.7)
        cb.set_label('meters')
        contour(fg.X, fg.Y, fg.B0, [0], colors='g')

        ##Loyce, CC is around latitude 41.
        gca().set_aspect(1./cos(38*pi/180.))
        ticklabel_format(useOffset=False)
        xticks(rotation=20)
        title('Maximum flow depth over %.2f hours' % t_hours)
        savefigp('h_onshore_' + run + '.png')
        # In the plot above, green shows fgmax points that never got wet.
        # The green contour shows `B0 = 0`.
        # White areas are masked out because they were initially wet.
        #
        # Regions colored blue/red/purple are initially dry fgmax points that
        # did not get wet during the tsunami, with color showing the maximum depth
        # of water recorded.

        ## Plot maximum speed

        bounds_speed = np.array([1e-6,0.5,1.0,1.5,2,2.5,3,4.5,6])
        cmap_speed = mpl.colors.ListedColormap([[.9,.9,1],[.6,.6,1],[.3,.3,1],[0,0,1],\
                                                [1,.8,.8],[1,.6,.6], [1,.3,.3], [1,0,0]])

        bounds_speed = np.array([1e-6,0.5,1.0,1.5,2,2.5,3,4.5])
        cmap_speed = mpl.colors.ListedColormap([[.9,.9,1],[.6,.6,1],[.3,.3,1],[0,0,1],\
                                                [1,.8,.8],[1,.6,.6], [1,0,0]])

        # Set color for value exceeding top of range to purple:
        cmap_speed.set_over(color=[1,0,1])

        # Set color for land points without inundation to light green:
        cmap_speed.set_under(color=[.7,1,.7])

        norm_speed = colors.BoundaryNorm(bounds_speed, cmap_speed.N)

        figure(figsize=(10,10))
        pc = plottools.pcolorcells(fg.X, fg.Y, fg.s, cmap=cmap_speed, norm=norm_speed)
        cb = colorbar(pc, extend='max', shrink=0.7)
        cb.set_label('m/s')
        contour(fg.X, fg.Y, fg.B0, [0], colors='g')
        gca().set_aspect(1./cos(38*pi/180.))
        ticklabel_format(useOffset=False)
        xticks(rotation=20)
        title('Maximum speed over %.2f hours' % t_hours)
        savefigp('speed_' + run + '.png')

        # The plot above shows the maximum speed at each fgmax point. The points colored
        # green remained dry over this simulation. The green contour shows `B0 = 0`.
        #
        # White areas are masked out because they were not fgmax points. Regions colored
        # blue or red are either offshore (initially wet) or onshore points that got wet,
        # colored by the maximum water speed $s = \sqrt{u^2 + v^2}$ over the simulation.

        ## Plot maximum eta offshore

        bounds_eta = array([0,0.5,1.0,1.5,2,2.5,3.0])

        cmap_eta=colors.ListedColormap([[.7,.7,1],[.5,.5,1],[0,0,1],[1,.7,.7],\
                                        [1,.4,.4],[1,0,0]])

        # Set color for value exceeding top of range to purple:
        cmap_eta.set_over(color=[1,0,1])

        norm_eta = colors.BoundaryNorm(bounds_eta, cmap_eta.N)

        figure(figsize=(10,10))
        pc = plottools.pcolorcells(fg.X, fg.Y, fg.eta_offshore, cmap=cmap_eta, norm=norm_eta)
        cb = colorbar(pc, extend='max', shrink=0.7)
        cb.set_label('meters')
        contour(fg.X, fg.Y, fg.B0, [0], colors='g')
        gca().set_aspect(1./cos(38*pi/180.))
        ticklabel_format(useOffset=False)
        xticks(rotation=20)
        title('Maximum offshore surface eta h+B0 over %.2f hours' % t_hours)
        savefigp('eta_offshore_' + run + '.png')

        ## Transect plots
        if fgno == 999: #set the fgno if you want transect plots plus the y-axis + x axis info
            y0_values = [37.8, 37.9, 38.0, 38.1, 38.2, 38.3 ]
            xmin = 140.875
            xmax = 141.08

            figure(figsize=(13,10))
            plotZ(fg.B0, show_cb=True)
            title('B0 (pre-seismic) with transects');

            for y0 in y0_values:
                plot([xmin,xmax], [y0,y0], 'k')

            savefigp('B0_with_transects_' + run + '.png')

            for y0 in y0_values:
                xout = linspace(xmin,xmax,1001)
                yout = y0*ones(xout.shape)
                B = gridtools.grid_eval_2d(fg.X, fg.Y, fg.B, xout, yout, return_ma=True)
                B = ma.masked_where(B < -1e90, B)
                B0 = gridtools.grid_eval_2d(fg.X, fg.Y, fg.B0, xout, yout, return_ma=True)
                B0 = ma.masked_where(B0 < -1e90, B0)
                h = gridtools.grid_eval_2d(fg.X, fg.Y, fg.h, xout, yout, return_ma=True)
                figure(figsize=(14,5))
                fill_between(xout, B, -100*ones(xout.shape), color=[.5,1,.5])
                fill_between(xout, B, B + h, color=[.5,.5,1])

                plot(xout, B + h, 'b', label='Post-seismic B + max h')

                plot(xout, B, 'g', label='Post-seismic B')
                plot(xout, B-B0, 'k--', label='Co-seismic displacement of z=0')
                legend(loc='upper right')
                xlim(xmin, xmax)
                ylim(-15, 20)
                ylabel('meters relative to MHW')
                grid(True)
                ##Need int and round to get the name correct below
                title('Post-seismic B + max h at latitude y = %.3f' % y0);
                savefigp('transect_y38-%3i.png' % int(round(1000*(y0-38))))

                ##Loyce:  added this figure with B0+h (eta) instead of B+h
                figure(figsize=(14,5))
                fill_between(xout, B0, -100*ones(xout.shape), color=[.5,1,.5])
                fill_between(xout, B0, B0 + h, color=[.5,.5,1])
                plot(xout, B0 + h, 'b', label='Eta, Pre-seismic B0 + max h')

                plot(xout, B0, 'g', label='Pre-seismic B')
                plot(xout, B-B0, 'k--', label='Co-seismic displacement of z=0')
                legend(loc='upper right')
                xlim(xmin,xmax)
                ylim(-15,20)
                ylabel('meters relative to MHW')
                grid(True)
                title('Eta, Pre-seismic B0 + max h at latitude y = %.3f' % y0);
                savefigp('transect_withB0_y38-%3i.png' % int(round(1000*(y0-38))))

        ######  Saving .npy h file
        ##      If fine, just fg.h, if coarse, replicate firs then save
        testing_npy = True
        hfile = 'h_Sedai_' + run + '_'+ dtopo.split('.')[0] +  '.npy'
        fname_hnpy = gis_dir + '/' + hfile
        save(fname_hnpy, array(fg.h))

        #Save x,y,B0,B,dz,eta,speed at point info file and as ascii raster file
        #point file

        pfile = 'p_Sedai_' + run + '_'+ dtopo.split('.')[0] + '.csv'
        pname_csv = gis_dir + '/' + pfile
        ptable = pd.DataFrame(list(zip(fg.X.flatten(), fg.Y.flatten(), fg.B0.flatten(), fg.B.flatten(), fg.dz.flatten(),
                                       fg.h.flatten(),fg.s.flatten())), columns=['X', 'Y','B0','B','dz','h','v'])
        ptable = ptable.round({'X':6, 'Y':6,'B0':3,'B':3,'dz':3,'d':3,'v':3})
        ptable = ptable.drop(ptable[ptable.B < 0].index)
        ptable.to_csv(pname_csv, sep=' ', header=True, index=None, mode='w')
        #asc files
        vars = ['B0','B','dz','h','v']
        Btable = pd.DataFrame(fg.B0).T[::-1]
        htables = [pd.DataFrame(fg.B0).T[::-1], pd.DataFrame(fg.B).T[::-1], pd.DataFrame(fg.dz).T[::-1],
                  pd.DataFrame(fg.h).T[::-1], pd.DataFrame(fg.s).T[::-1]]
        i = 0
        for var in vars:
            print('processing parameter ', (i+1), ':', var)
            htable = htables[i]
            i = i+1
            ascfile = 'Sedai_' + var + '_'+ run + '_'+ dtopo.split('.')[0] + '.acs'
            prjname = 'Sedai_' + var + '_'+ run + '_'+ dtopo.split('.')[0] + '.prj'
            afname = gis_dir + '/' + ascfile
            prj_fname= gis_dir + '/' + prjname
            if var == 'h':
                htable[Btable < 0] = -32768
                htable[htable == 0] = -32768

            num_rows, num_cols = htable.shape
            xll = np.amin(fg.X)
            yll = np.amin(fg.Y)
            ascfname = open(afname, "w")
            ascfname.write('ncols ' + str(num_cols) + '\n')
            ascfname.write('nrows ' + str(num_rows) + '\n')
            ascfname.write('xllcorner ' + str(xll) + '\n')
            ascfname.write('yllcorner ' + str(yll) + '\n')
            ascfname.write('cellsize ' + str(res) + '\n')
            ascfname.write('nodata_value -32768' + '\n')
            ascfname.close()
            htable.to_csv(afname, sep=' ', header=False, index=None, mode='a')

            prjline = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],' \
                      'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
            with open(prj_fname, 'w') as f:
                f.write(prjline)

