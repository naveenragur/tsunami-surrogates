
"""
Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.

"""
import numpy as np, os

try:
    root_dir = os.environ['PTHA']
except:
    raise Exception("*** Must first set PTHA enviornment variable")

try:
    D21401 = np.loadtxt(root_dir + '/gis/2011/Formatted/21401_notide.txt')
    D21413 = np.loadtxt(root_dir + '/gis/2011/Formatted/21413_notide.txt')
    D21418 = np.loadtxt(root_dir + '/gis/2011/Formatted/21418_notide.txt')
    D21419 = np.loadtxt(root_dir + '/gis/2011/Formatted/21419_notide.txt')
    JMAR = np.loadtxt(root_dir + '/gis/2011/Formatted/JMAR_10001_5.txt')
    JMA = np.loadtxt(root_dir + '/gis/2011/Formatted/JMA_205_801_806.txt')
except:
    print("*** Could not load observation data files")
#------------------
def setplot(plotdata):
#--------------------------
    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of clawpack.visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.

    """
    from clawpack.visclaw import colormaps, geoplot
    from numpy import linspace

    plotdata.clearfigures()  # clear any old figures,axes,items data


    # To plot gauge locations on pcolor or contour plot, use this as
    # an afteraxis function:

    def addgauges(current_data):
        from clawpack.visclaw import gaugetools
        gaugetools.plot_gauge_locations(current_data.plotdata,\
             gaugenos = 'all', format_string='ko', add_labels=True)

    def fixup(current_data):
        import pylab
        addgauges(current_data)
        t = current_data.t
        t = t / 3600.  # hours
        pylab.title('Surface at %4.2f hours' % t, fontsize=20)
        #pylab.xticks(fontsize=15)
        #pylab.yticks(fontsize=15)

    def fixupnogauge(current_data):
        import pylab
        #addgauges(current_data)
        t = current_data.t
        t = t / 3600.  # hours
        pylab.title('Surface at %4.2f hours' % t, fontsize=20)
        #pylab.xticks(fontsize=15)
        #pylab.yticks(fontsize=15)


    #-----------------------------------------
    # Figure for imshow plot
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Domain', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes('imshow')
    plotaxes.title = 'Surface'
    plotaxes.scaled = True

    # plotaxes.afteraxes = fixup #to add gague on map plot

    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    # plotitem.plot_var = geoplot.surface
    plotitem.plot_var = geoplot.surface_or_depth
    plotitem.imshow_cmap = geoplot.tsunami_colormap
    plotitem.imshow_cmin = -0.5
    plotitem.imshow_cmax = 0.5
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.patchedges_show = 0
    #plotitem.amr_patchedges_show = [1,1,1,0,0]  # only coarse levels

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.land
    plotitem.imshow_cmap = geoplot.land_colors
    plotitem.imshow_cmin = 0.0
    plotitem.imshow_cmax = 100.0
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.patchedges_show = 0
    #plotitem.amr_patchedges_show = [1,1,1,0,0]  # only coarse levels
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'

    # add contour lines of bathy if desired:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    plotitem.contour_levels = linspace(-2000,0,5)
    plotitem.amr_contour_colors = ['y']  # color on each level
    plotitem.kwargs = {'linestyles':'solid','linewidths':2}
    plotitem.amr_contour_show = [1,0,0]
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0


    #-----------------------------------------
    # Figure for zoom plot
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Honshu', figno=2)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes('imshow')
    plotaxes.title = 'Surface'
    plotaxes.scaled = True

    # plotaxes.afteraxes = fixup

    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    # plotitem.plot_var = geoplot.surface
    plotitem.plot_var = geoplot.surface_or_depth
    plotitem.imshow_cmap = geoplot.tsunami_colormap
    plotitem.imshow_cmin = -1.
    plotitem.imshow_cmax = 1.
    plotitem.add_colorbar = True
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.patchedges_show = 0

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.land
    plotitem.imshow_cmap = geoplot.land_colors
    plotitem.imshow_cmin = 0.0
    plotitem.imshow_cmax = 100.0
    plotitem.add_colorbar = False
    plotitem.amr_celledges_show = [0,0,0]
    plotitem.patchedges_show = 0
    plotaxes.xlimits = [139., 145.]
    plotaxes.ylimits = [34., 42.]

    # add contour lines of bathy if desired:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    plotitem.contour_levels = linspace(-2000,0,5)
    plotitem.amr_contour_colors = ['y']  # color on each level
    plotitem.kwargs = {'linestyles':'solid','linewidths':2}
    plotitem.amr_contour_show = [1,0,0]
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0


    #-----------------------------------------
    # Figure for zoom plot
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Sendai Bay', figno=3)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes('imshow')
    plotaxes.title = 'Surface'
    plotaxes.scaled = True

    # plotaxes.afteraxes = fixupnogauge

    # Water
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    # plotitem.plot_var = geoplot.surface
    plotitem.plot_var = geoplot.surface_or_depth
    plotitem.imshow_cmap = geoplot.tsunami_colormap
    plotitem.imshow_cmin = -0.2
    plotitem.imshow_cmax = 0.2
    plotitem.add_colorbar = True
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0

    # Land
    plotitem = plotaxes.new_plotitem(plot_type='2d_imshow')
    plotitem.plot_var = geoplot.land
    plotitem.imshow_cmap = geoplot.land_colors
    plotitem.imshow_cmin = 0.0
    plotitem.imshow_cmax = 10.0
    plotitem.add_colorbar = False
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0
    plotaxes.xlimits = [140.72, 141.76]
    plotaxes.ylimits = [37.67, 38.68]

    # add contour lines of bathy if desired:
    plotitem = plotaxes.new_plotitem(plot_type='2d_contour')
    plotitem.show = False
    plotitem.plot_var = geoplot.topo
    #plotitem.contour_levels = linspace(-2000,0,5)
    plotitem.contour_levels = linspace(0,8,9)
    plotitem.amr_contour_colors = ['y']  # color on each level
    plotitem.kwargs = {'linestyles':'solid','linewidths':2}
    plotitem.amr_contour_show = [0,0,0,0,0,1]
    plotitem.celledges_show = 0
    plotitem.patchedges_show = 0


    #-----------------------------------------
    # Figures for gauges
    #-----------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Surface', figno=300, \
                    type='each_gauge')
    plotfigure.clf_each_gauge = True

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    #plotaxes.axescmd = 'subplot(2,1,1)'
    plotaxes.title = 'Surface'

    # Plot surface as red curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 3
    plotitem.plotstyle = 'r-'
    plotitem.kwargs = {'linewidth':2}

    # Plot topo as green curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.show = False

    def gaugetopo(current_data):
        q = current_data.q
        h = q[0,:]
        eta = q[3,:]
        topo = eta - h
        dz = max(topo)-min(topo)
        return eta + dz

    plotitem.plot_var = gaugetopo
    plotitem.plotstyle = 'g--'

    def add_zeroline(current_data):
        from pylab import plot, legend, xticks, floor, xlim,ylim,axis, xlabel
        t = current_data.t
        q = current_data.q
        topo = q[3,:] - q[0,:]
        gaugeno = current_data.gaugeno
        #Time,21401, 21413, 21418, 21419
        if gaugeno == 21401:
            plot(D21401[:,0], D21401[:,1], 'k')
        if gaugeno == 21413:
            plot(D21413[:,0], D21413[:,1], 'k')
        if gaugeno == 21418:
            plot(D21418[:,0], D21418[:,1], 'k')
        if gaugeno == 21419:
            plot(D21419[:,0], D21419[:,1], 'k')
        #Time,205, 801, 802, 803, 806
        elif gaugeno == 205:
            plot(JMA[:, 0], JMA[:, 1], 'k')
        elif gaugeno == 801:
            plot(JMA[:, 0], JMA[:, 2], 'k')
        elif gaugeno == 802:
            plot(JMA[:, 0], JMA[:, 3], 'k')
        elif gaugeno == 803:
            plot(JMA[:, 0], JMA[:, 4], 'k')
        elif gaugeno == 804:
            plot(JMA[:, 0], JMA[:, 5], 'k')
        elif gaugeno == 806:
            plot(JMA[:, 0], JMA[:, 6], 'k')
        #Time,10001, 10002, 10003, 10004, 10005
        elif gaugeno == 10001:
            plot(JMAR[0, 0]*3600, JMAR[0, 1] ,'ro', markersize=10)
            plot(JMAR[0, 2] * 3600, JMAR[0, 3], 'ko', markersize=10)
        elif gaugeno == 10002:
            plot(JMAR[1, 0]*3600, JMAR[1, 1], 'ro', markersize=10)
            plot(JMAR[1, 2] * 3600, JMAR[1, 3] ,'ko', markersize=10)
        elif gaugeno == 10003:
            plot(JMAR[2, 0]*3600, JMAR[2, 1], 'ro', markersize=10)
            plot(JMAR[1, 2] * 3600, JMAR[1, 3] ,'ko', markersize=10)
        elif gaugeno == 10004:
            plot(JMAR[3, 0]*3600, JMAR[3, 1], 'ro', markersize=10)
            plot(JMAR[3, 2] * 3600, JMAR[3, 3], 'ko', markersize=10)
        elif gaugeno == 10005:
            plot(JMAR[4, 0]*3600, JMAR[4, 1], 'ro', markersize=10)
            plot(JMAR[4, 2] * 3600, JMAR[4, 3], 'ko', markersize=10)

        plot(t, 0*t, 'k')
        n = int(floor(t.max()/3600.) + 2)
        xticks([3600*i for i in range(n)], ['%i' % i for i in range(n)])
        xlabel('time (hours)')
        print("+++ gaugeno = ", current_data.gaugeno)

    def add_legend_eta(current_data):
        from pylab import legend
        legend(('Surface'),loc='lower left')
        add_zeroline(current_data)

    # def ylim(current_data):
    #     if current_data.gaugeno == 9001:
    #         gauge_ylimits = [-2, 2]
    #     elif current_data.gaugeno == 7021:
    #         gauge_ylimits = [-5, 5]
    #     elif current_data.gaugeno == 7020 or current_data.gaugeno == 7019 or current_data.gaugeno == 7018:
    #         gauge_ylimits = [-8, 8]
    #     elif current_data.gaugeno == 8001 or current_data.gaugeno == 8002 or current_data.gaugeno == 8003 \
    #             or current_data.gaugeno == 8004 or current_data.gaugeno == 8005:
    #         gauge_ylimits = [-10, 10]
    #     return gauge_ylimits
    # plotaxes.ylimits = ylim  #
    #plotaxes.ylimits = [-8, 8]
    plotaxes.xlimits = [0, 10800]  # First 3 hrs
    plotaxes.afteraxes = add_zeroline
#----------------------------------------------
    plotfigure = plotdata.new_plotfigure(name='Velocities', figno=301, \
                    type='each_gauge')
    plotfigure.clf_each_gauge = True

    plotaxes = plotfigure.new_plotaxes()
    #plotaxes.axescmd = 'subplot(2,1,2)'
    plotaxes.title = 'Velocities'
    plotaxes.afteraxes = add_zeroline

    # Plot velocity as red curve:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.show = True
    def speed(current_data):
        from numpy import where, sqrt
        h = current_data.q[0,:]
        h = where(h>0.01, h, 1.e6)
        u = 100. * current_data.q[1,:] / h
        v = 100. * current_data.q[2,:] / h
        s = sqrt(u**2 + v**2)
        return s
    plotitem.plot_var = speed
    plotitem.plotstyle = 'k-'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def uvel(current_data):
        from numpy import where, sqrt
        h = current_data.q[0,:]
        h = where(h>0.01, h, 1.e6)
        u = 100. * current_data.q[1,:] / h
        return u
    plotitem.plot_var = uvel
    plotitem.plotstyle = 'r-'
    plotitem.kwargs = {'linewidth':2}

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def vvel(current_data):
        from numpy import where, sqrt
        h = current_data.q[0,:]
        h = where(h>0.01, h, 1.e6)
        v = 100. * current_data.q[2,:] / h
        return v
    plotitem.plot_var = vvel
    plotitem.plotstyle = 'g-'
    plotitem.kwargs = {'linewidth':2}

    def add_legend_vel(current_data):
        from pylab import legend
        # legend(["u","v"],'upper left')
        legend(['Speed','uvel','vvel'],loc='upper left')
        add_zeroline(current_data)

    #plotaxes.ylimits = [-500,500]
    plotaxes.afteraxes = add_legend_vel


    #-----------------------------------------
    # Plots of timing (CPU and wall time):

    def make_timing_plots(plotdata):
        from clawpack.visclaw import plot_timing_stats
        import os,sys
        try:
            timing_plotdir = plotdata.plotdir + '/_timing_figures'
            os.system('mkdir -p %s' % timing_plotdir)
            # adjust units for plots based on problem:
            units = {'comptime':'seconds', 'simtime':'hours', 
                     'cell':'millions'}
            plot_timing_stats.make_plots(outdir=plotdata.outdir, 
                                          make_pngs=True,
                                          plotdir=timing_plotdir, 
                                          units=units)
        except:
            print('*** Error making timing plots')

    otherfigure = plotdata.new_otherfigure(name='timing plots',
                    fname='_timing_figures/timing.html')
    otherfigure.makefig = make_timing_plots



    #-----------------------------------------

    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via clawpack.visclaw.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'         # list of frames to print #control for animation 'all' or empty
    plotdata.print_gaugenos = [5832,6042]          # list of gauges to print
    plotdata.print_fignos = [1,2,3,300,301]   # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = False                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?
    plotdata.parallel = True

    return plotdata
