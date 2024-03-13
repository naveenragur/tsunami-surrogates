"""
Module to set up run time parameters for Clawpack -- AMRClaw code.

The values set in the function setrun are then written out to data files in the rundir location mentioned below
that will be read in by the Fortran code.

"""

import os, csv
import numpy as np
from clawpack.amrclaw import region_tools
from clawpack.geoclaw import fgmax_tools
from clawpack.amrclaw.data import FlagRegion

try:
    CLAW = os.environ['CLAW']
    HOME = os.environ['HOME']
except:
    raise Exception("*** Must first set CLAW environment variable")

# Environment variable PTHA should be set to top level of this repository

try:
    root_dir = os.environ['PTHA']
except:
    raise Exception("*** Must first set PTHA enviornment variable")

topodir = os.path.join(root_dir,'gis','topo','ASC')
dtopodir = os.path.join(root_dir,'_tsunami')
if not os.path.isdir(topodir):
    raise Exception("*** Missing directory: %s" % topodir)

# initial run without a tsunami forcing to derive bathymetry
making_B0 = False
making_DART = True
making_inundation = True
print("Simulation for with no displacement :", making_B0, " and DART buoys :", making_DART)

# ------------------------------
def setrun(claw_pkg='geoclaw'):
    # ------------------------------

    """
    Define the parameters used for running Clawpack.

    INPUT:
        claw_pkg expected to be "geoclaw" for this setrun.

    OUTPUT:
        rundata - object of class ClawRunData

    """

    from clawpack.clawutil import data
    assert claw_pkg.lower() == 'geoclaw', "Expected claw_pkg = 'geoclaw'"

    num_dim = 2
    rundata = data.ClawRunData(claw_pkg, num_dim)

    # ------------------------------------------------------------------
    # Problem-specific parameters to be written to setprob.data:
    # ------------------------------------------------------------------
    # probdata = rundata.new_UserData(name='probdata',fname='setprob.data')

    # ------------------------------------------------------------------
    # GeoClaw specific parameters:
    # ------------------------------------------------------------------
    rundata = setgeo(rundata)

    # ------------------------------------------------------------------
    # Standard Clawpack parameters to be written to claw.data:
    # ------------------------------------------------------------------

    clawdata = rundata.clawdata  # initialized when rundata instantiated

    # Set single grid parameters first.
    # See below for AMR parameters.

    # ---------------
    # Spatial domain:
    # ---------------

    # Number of space dimensions:
    clawdata.num_dim = num_dim

    if making_DART:
        # Lower and upper edge of computational domain: DART
        clawdata.lower[0] = 139.0 # xlower
        clawdata.upper[0] = 156.0 # xupper
        clawdata.lower[1] = 29.0  # ylower
        clawdata.upper[1] = 45.0  # yupper
    else:
        # Lower and upper edge of computational domain:DTOPO
        clawdata.lower[0] = 139.0  # xlower
        clawdata.upper[0] = 145.0  # xupper
        clawdata.lower[1] = 34.0  # ylower
        clawdata.upper[1] = 42.0  # yupper
    
    # Res of raw data {"2700":.02430, "1350": 0.01215, "0450" : 0.00405, "0150" : 0.00135, "0050": 0.00045}
    # 2700  = 2*0.01215/2 level 1
    # 1350  = 0.01215/3 level 2
    # 0450  = 0.00405/3 level 3
    # 0150  = 0.00135/3 level 4
    # 0050  = 0.00045   level 5

    clawdata.num_cells[0] = round((clawdata.upper[0] - clawdata.lower[0]) / (0.01215*2))  # mx
    clawdata.num_cells[1] = round((clawdata.upper[1] - clawdata.lower[1]) / (0.01215*2))  # my

    print('The main domain has mx,my:',clawdata.num_cells,'dx,dy:', (0.01215*2))

    # ---------------
    # Size of system:
    # ---------------

    # Number of equations in the system:
    clawdata.num_eqn = 3

    # Number of auxiliary variables in the aux array (initialized in setaux)
    clawdata.num_aux = 3

    # Index of aux array corresponding to capacity function, if there is one:
    clawdata.capa_index = 2

    # -------------
    # Initial time:
    # -------------

    clawdata.t0 = 0.0

    # Restart from checkpoint file of a previous run?
    # Note: If restarting, you must also change the Makefile to set:
    #    RESTART = True
    # If restarting, t0 above should be from original run, and the
    # restart_file 'fort.chkNNNNN' specified below should be in
    # the OUTDIR indicated in Makefile.

    clawdata.restart = False  # True to restart from prior results
    clawdata.restart_file = 'fort.chk00000'  # File to use for restart data

    # -------------
    # Output times:
    # --------------

    # Specify at what times the results should be written to fort.q files.
    # Note that the time integration stops after the final output time.

    clawdata.output_style = 1

    if clawdata.output_style == 1:
        # Output ntimes frames at equally spaced times up to tfinal:
        # Can specify num_output_times = 0 for no output
        if making_B0:
            clawdata.num_output_times = 1  # for making B0
            clawdata.tfinal = 1.  # for making B0
            clawdata.output_t0 = True  # for making B0
        else:
            clawdata.num_output_times = 1
            clawdata.tfinal = 1#6 * 3600.
            clawdata.output_t0 = True  # output at initial (or restart) time?

    elif clawdata.output_style == 2:
        # Specify a list or numpy array of output times:
        # Include t0 if you want output at the initial time.
        clawdata.output_times = np.linspace(7, 13, 4) * 3600.

    elif clawdata.output_style == 3:
        # Output every step_interval timesteps over total_steps timesteps:
        clawdata.output_step_interval = 1
        clawdata.total_steps = 1
        clawdata.output_t0 = False  # output at initial (or restart) time?

    clawdata.output_format == 'binary'  # 'ascii', 'binary', 'netcdf'

    clawdata.output_q_components = 'all'  # could be list such as [True,True]
    clawdata.output_aux_components = 'none'  # could be list
    clawdata.output_aux_onlyonce = True  # output aux arrays only at t0

    # ---------------------------------------------------
    # Verbosity of messages to screen during integration:
    # ---------------------------------------------------

    # The current t, dt, and cfl will be printed every time step
    # at AMR levels <= verbosity.  Set verbosity = 0 for no printing.
    #   (E.g. verbosity == 2 means print only on levels 1 and 2.)
    clawdata.verbosity = 0

    # --------------
    # Time stepping:
    # --------------

    # if dt_variable==True:  variable time steps used based on cfl_desired,
    # if dt_variable==False  Fixed time steps dt = dt_initial always used.
    clawdata.dt_variable = True

    # Initial time step for variable dt.
    # (If dt_variable==0 then dt=dt_initial for all steps)
    clawdata.dt_initial = 1

    # Max time step to be allowed if variable dt used:
    clawdata.dt_max = 1e+99

    # Desired Courant number if variable dt used
    clawdata.cfl_desired = 0.8
    # max Courant number to allow without retaking step with a smaller dt:
    clawdata.cfl_max = 1.0

    # Maximum number of time steps to allow between output times:
    clawdata.steps_max = 50000

    # ------------------
    # Method to be used:
    # ------------------

    # Order of accuracy:  1 => Godunov,  2 => Lax-Wendroff plus limiters
    clawdata.order = 2

    # Use dimensional splitting? (not yet available for AMR)
    clawdata.dimensional_split = 'unsplit'

    # For unsplit method, transverse_waves can be
    #  0 or 'none'      ==> donor cell (only normal solver used)
    #  1 or 'increment' ==> corner transport of waves
    #  2 or 'all'       ==> corner transport of 2nd order corrections too
    clawdata.transverse_waves = 2

    # Number of waves in the Riemann solution:
    clawdata.num_waves = 3

    # List of limiters to use for each wave family:
    # Required:  len(limiter) == num_waves
    # Some options:
    #   0 or 'none'     ==> no limiter (Lax-Wendroff)
    #   1 or 'minmod'   ==> minmod
    #   2 or 'superbee' ==> superbee
    #   3 or 'mc'       ==> MC limiter
    #   4 or 'vanleer'  ==> van Leer
    clawdata.limiter = ['vanleer', 'vanleer', 'vanleer']

    clawdata.use_fwaves = True  # True ==> use f-wave version of algorithms

    # Source terms splitting:
    #   src_split == 0 or 'none'    ==> no source term (src routine never called)
    #   src_split == 1 or 'godunov' ==> Godunov (1st order) splitting used,
    #   src_split == 2 or 'strang'  ==> Strang (2nd order) splitting used,  not recommended.
    clawdata.source_split = 'godunov'

    # --------------------
    # Boundary conditions:
    # --------------------

    # Number of ghost cells (usually 2)
    clawdata.num_ghost = 2

    # Choice of BCs at xlower and xupper:
    #   0 or 'user'     => user specified (must modify bcNamr.f to use this option)
    #   1 or 'extrap'   => extrapolation (non-reflecting outflow)
    #   2 or 'periodic' => periodic (must specify this at both boundaries)
    #   3 or 'wall'     => solid wall for systems where q(2) is normal velocity

    clawdata.bc_lower[0] = 'extrap'  # at xlower
    clawdata.bc_upper[0] = 'extrap'  # at xupper

    clawdata.bc_lower[1] = 'extrap'  # at ylower
    clawdata.bc_upper[1] = 'extrap'  # at yupper

    # ---------------
    # Gauges:
    # ---------------
    # gauges = rundata.gaugedata.gauges

    # for gauges append lines of the form [gaugeno, x, y, t1, t2]

    rundata.gaugedata.gtype = 'stationary'
    rundata.gaugedata.min_time_increment = 1.  # seconds between gauge output
    rundata.gaugedata.display_format = "f15.7"  # need enough digits for u,v

    tstart_gauges = 0 * 60.  # start at time 0.

    # list of possible real gauges:
    gauges_csv_file = root_dir + '/gis/2011/Formatted/ObservationLocationInfo.csv'
    print(gauges_csv_file)
    gauge_list = [205, 801, 802, 803, 804, 806, 10001, 10002, 10003, 10004, 10005] #
    if making_DART:
        gauge_list = [21401, 21413, 21418, 21419] + gauge_list

    # gauges to always include
    f = open(gauges_csv_file, 'r')
    csv_reader = csv.reader(f)
    gaugeloc = {}
    gaugex = {}
    gaugey = {}
    for row in csv_reader:
        try:
            xg = float(row[3])
            yg = float(row[2])
        except:
            continue  # go to next row
        gaugeno = int(row[4])
        gaugeloc[gaugeno] = row[4]
        gaugex[gaugeno] = xg
        gaugey[gaugeno] = yg
    f.close()

    for gaugeno in gauge_list:
        rundata.gaugedata.gauges.append([gaugeno, gaugex[gaugeno], gaugey[gaugeno], \
                                         tstart_gauges, 1e9])
        print('Including gauge %3i at %s' % (gaugeno, gaugeloc[gaugeno]))

    # list of propagation observation gauges:
    pgauges_csv_file = root_dir + '/gis/coastlines/prop_gauge_pts.csv'
    print(pgauges_csv_file)

    f = open(pgauges_csv_file, 'r')
    csv_reader = csv.reader(f)
    gauge_list = {}
    gaugeloc = {}
    gaugex = {}
    gaugey = {}
    for row in csv_reader:
        try:
            xg = float(row[3])
            yg = float(row[2])
        except:
            continue  # go to next row
        gaugeno = int(row[4])
        gauge_list[gaugeno] = int(row[4])
        gaugeloc[gaugeno] = row[4]
        gaugex[gaugeno] = xg
        gaugey[gaugeno] = yg
    f.close()

    for gaugeno in gauge_list:
        rundata.gaugedata.gauges.append([gaugeno, gaugex[gaugeno], gaugey[gaugeno], \
                                         tstart_gauges, 1e9])
        print('Including gauge %3i at %s' % (gaugeno, gaugeloc[gaugeno]))

    # --------------
    # Checkpointing:
    # --------------

    # Specify when checkpoint files should be created that can be
    # used to restart a computation.

    clawdata.checkpt_style = 0

    if clawdata.checkpt_style == 0:
        # Do not checkpoint at all
        pass

    elif clawdata.checkpt_style == 1:
        # Checkpoint only at tfinal.
        pass

    elif clawdata.checkpt_style == 2:
        # Specify a list of checkpoint times.
        clawdata.checkpt_times = [0.1, 0.15]

    elif clawdata.checkpt_style == 3:
        # Checkpoint every checkpt_interval timesteps (on Level 1)
        # and at the final time.
        clawdata.checkpt_interval = 5

    # ---------------
    # AMR parameters:   (written to amr.data)
    # ---------------
    amrdata = rundata.amrdata
    # amrdata.memsize = 2**28

    # max number of refinement levels:
    if making_DART:
        amrdata.amr_levels_max = 4  # Set to max value for 5
    elif making_inundation:
        amrdata.amr_levels_max = 5  # Set to max value for 5
    else:
        amrdata.amr_levels_max = 3  # Set to max value for 5

    # Res of raw data {"2700":.02430, "1350": 0.01215, "0450" : 0.00405, "0150" : 0.00135, "0050": 0.00045}
    # 2700  = 2*0.01215/2 level 1
    # 1350  = 0.01215/3 level 2
    # 0450  = 0.00405/3 level 3
    # 0150  = 0.00135/3 level 4
    # 0050  = 0.00045   level 5

    # List of refinement ratios at each level (length at least amr_level_max-1)
    amrdata.refinement_ratios_x = [2, 3, 3, 3]
    amrdata.refinement_ratios_y = [2, 3, 3, 3]
    amrdata.refinement_ratios_t = [2, 3, 3, 3]

    # Specify type of each aux variable in amrdata.auxtype.
    # This must be a list of length num_aux, each element of which is one of:
    #   'center',  'capacity', 'xleft', or 'yleft'  (see documentation).
    amrdata.aux_type = ['center', 'capacity', 'yleft']

    # Flag for refinement based on Richardson error estimater:
    amrdata.flag_richardson = False  # use Richardson?
    amrdata.flag_richardson_tol = 1.0  # Richardson tolerance

    # Flag for refinement using routine flag2refine:
    amrdata.flag2refine = True  # use this?
    amrdata.flag2refine_tol = 0.5  # tolerance used in this routine
    # Note: in geoclaw the refinement tolerance is set as wave_tolerance below
    # and flag2refine_tol is unused!

    # steps to take on each level L between regriddings of level L+1:
    amrdata.regrid_interval = 4

    # width of buffer zone around flagged points:
    # (typically the same as regrid_interval so waves don't escape):
    amrdata.regrid_buffer_width = 4

    # clustering alg. cutoff for (# flagged pts) / (total # of cells refined)
    # (closer to 1.0 => more small grids may be needed to cover flagged cells)
    amrdata.clustering_cutoff = 0.7

    # print info about each regridding up to this level:
    amrdata.verbosity_regrid = 1

    # ---------------
    # Regions:
    # ---------------
    regions = rundata.regiondata.regions
    # to specify regions of refinement append lines of the form
    # [minlevel,maxlevel,t1,t2,x1,x2,y1,y2]

    # Res of raw data {"2700":.02430, "1350": 0.01215, "0450" : 0.00405, "0150" : 0.00135, "0050": 0.00045}
    # 2700  = 2*0.01215/2 level 1
    # 1350  = m at depths arou/2 level 2
    # 0450  = 0.00405/3 level 3
    # 0150  = 0.00135/3 level 4
    # 0050  = 0.00045   level 5

    flagregions = rundata.flagregiondata.flagregions
    # Region 01 : Global region.  This assures a maximum refinement in any
    # regions not covered by other regions listed below when waves travel at deep depth
    flagregion = FlagRegion(num_dim=2)
    flagregion.name = 'Region01_domain'
    flagregion.minlevel = 1
    flagregion.maxlevel = 2
    flagregion.t1 = 0.
    flagregion.t2 = 1e9
    flagregion.spatial_region_type = 1  # Rectangle
    flagregion.spatial_region = [clawdata.lower[0] - 0.1,
                                 clawdata.upper[0] + 0.1,
                                 clawdata.lower[1] - 0.1,
                                 clawdata.upper[1] + 0.1]
    flagregions.append(flagregion)

    # Region 02 : (region for the dtopo file)
    # Time interval taken from dtopo files : 300 for Satake dynamic event
    source_region = [139., 145., 34., 42.]
    flagregion = FlagRegion(num_dim=2)
    flagregion.name = 'Region_dtopo'
    flagregion.minlevel = 2
    flagregion.maxlevel = 2
    flagregion.t1 = 0.
    flagregion.t2 = 300
    flagregion.spatial_region_type = 1  # Rectangle
    flagregion.spatial_region = source_region
    flagregions.append(flagregion)

    # Region 03 : Region encompassing deeper Tohoku coast(100m-850m) ~ 450m resolution
    # Time interval :  (0,1e9)
    slu = np.array([[36.78, 140.69, 141.73],
                    [37.13, 140.84, 141.99],
                    [37.55, 140.85, 142.49],
                    [38.19, 140.72, 142.38],
                    [38.64, 141.20, 142.52],
                    [39.30, 141.70, 142.47],
                    [39.76, 141.96, 142.70]])

    rr = region_tools.RuledRectangle(slu=slu)
    rr.ixy = 'y'
    rr.method = 1
    rr_name = 'Region_100m_850m'
    rr.write(rr_name + '.data')
    rr.make_kml(fname=rr_name + '.kml', name=rr_name)

    flagregion = FlagRegion(num_dim=2)
    flagregion.name = 'Region_100m_850m'
    flagregion.minlevel = 3
    flagregion.maxlevel = 3
    flagregion.t1 = 0.
    flagregion.t2 = 1e9
    flagregion.spatial_region_type = 2  # RuledRectangle
    flagregion.spatial_region_file = root_dir + '/_input/Region_100m_850m.data'
    flagregions.append(flagregion)

    # Region 04 : Region encompassing shallower Tohoku coast (< 100m till coast) ~ 150m resolution
    # Time interval :  (0,1e9)
    slu = np.array([[36.85, 140.75, 141.12],
                    [37.15, 140.90, 141.23],
                    [37.725,140.90, 141.44],
                    [38.06, 140.83, 141.50],
                    [38.264,140.88, 141.69],
                    [38.625, 141.26, 141.68],
                    [38.78, 141.40, 141.76],
                    [38.99, 141.56, 141.94],
                    [39.36, 141.78, 142.10]])

    rr = region_tools.RuledRectangle(slu=slu)
    rr.ixy = 'y'
    rr.method = 1
    rr_name = 'Region_0m_100m'
    rr.write(rr_name + '.data')
    rr.make_kml(fname=rr_name + '.kml', name=rr_name)

    flagregion = FlagRegion(num_dim=2)
    flagregion.name = 'Region_0m_100m'
    flagregion.minlevel = 4
    flagregion.maxlevel = 4
    flagregion.t1 = 0.
    flagregion.t2 = 1e9
    flagregion.spatial_region_type = 2  # RuledRectangle
    flagregion.spatial_region_file = root_dir + '/_input/Region_0m_100m.data'
    flagregions.append(flagregion)

    # Region 05 : Region encompassing S.Sendai coast for fgmax readings using 50m for innundation.
    # Time interval :  (0,1e9)
    regions.append([5, 5, 0., 1e9, 140.84, 141.10, 37.75, 38.34]) #sendai
    regions.append([5, 5, 0., 1e9, 141.14, 141.39, 38.35, 38.49]) #ishinomaki
    regions.append([5, 5, 0., 1e9, 141.59, 141.69, 38.96, 39.04]) #rikuzentakata

    #  ----- For developers -----
    # Toggle debugging print statements:
    amrdata.dprint = False  # print domain flags
    amrdata.eprint = False  # print err est flags
    amrdata.edebug = False  # even more err est flags
    amrdata.gprint = False  # grid bisection/clustering
    amrdata.nprint = False  # proper nesting output
    amrdata.pprint = False  # proj. of tagged points
    amrdata.rprint = False  # print regridding summary
    amrdata.sprint = False  # space/memory output
    amrdata.tprint = False  # time step reporting each level
    amrdata.uprint = False  # update/upbnd reporting

    return rundata

    # end of function setrun
    # ----------------------


# -------------------
def setgeo(rundata):
    # -------------------
    """
    Set GeoClaw specific runtime parameters.
    """

    try:
        geo_data = rundata.geo_data
    except:
        print("*** Error, this rundata has no geo_data attribute")
        raise AttributeError("Missing geo_data attribute")

    # == Physics ==
    geo_data.gravity = 9.81
    geo_data.coordinate_system = 2
    geo_data.earth_radius = 6367500.0

    # == Forcing Options
    geo_data.coriolis_forcing = False

    # == Algorithm and Initial Conditions ==
    geo_data.sea_level = 0.0
    geo_data.dry_tolerance = 0.001
    geo_data.friction_forcing = True
    geo_data.manning_coefficient = 0.035
    geo_data.friction_depth = 500

    # Refinement settings
    refinement_data = rundata.refinement_data
    refinement_data.variable_dt_refinement_ratios = True
    refinement_data.wave_tolerance = 0.02

    # == settopo.data values ==
    # [topotype, fname]

    topofiles = rundata.topo_data.topofiles
    if making_DART:
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_gebco.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0150-08.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0150-09.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0150-10.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0050-20.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0050-21.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0050-22.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0050-23.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0050-24.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0050-25.asc')])
    else:
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_1350-01.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0450-03.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0450-04.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0150-08.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0150-09.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_wgs_0150-10.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_COP30_0050-20.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_COP30_0050-21.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_COP30_0050-22.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_COP30_0050-23.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_COP30_0050-24.asc')])
        topofiles.append([3, os.path.join(topodir, 'depth_COP30_0050-25.asc')])


    # == setdtopo.data values ==
    # [dtype, minlevel, maxlevel, 'dtopofname']
    # for moving topography, append lines of the form :   (<= 1 allowed for now!)

    if making_B0:
        rundata.dtopo_data.dtopofiles = []  # no dtopo for making B0
        rundata.dtopo_data.dt_max_dtopo = 1.0
    else:
        # Region 1, above handles this region.
        rundata.dtopo_data.dtopofiles = [[3, os.path.join(dtopodir, 'TEST.tt3')]]

    # == setqinit.data values ==
    rundata.qinit_data.qinit_type = 0
    rundata.qinit_data.qinitfiles = []

    if making_B0:
        rundata.fgmax_data.num_fgmax_val = 1  # for making B0 only depth

    else:
        rundata.fgmax_data.num_fgmax_val = 2  # Save depth and speed

    fgmax_grids = rundata.fgmax_data.fgmax_grids  # empty list to start

    # Now append to this list objects of class fgmax_tools.FGmaxGrid
    # specifying any fgmax grids.
    if making_inundation:
        tstart_max = 0.  # for making max depth
        dx150 = 0.00135
        dx050 = 0.00045

        ###### Sendai Bay
        # Points on a uniform 2d grid:
        fg = fgmax_tools.FGmaxGrid()
        fg.point_style = 2  # uniform rectangular x-y grid
        fg.x1 = 140.85 + dx050 / 2.0
        fg.x2 = 141.09 - dx050 / 2.0
        fg.y1 = 37.76 + dx050 / 2.0
        fg.y2 = 38.33 - dx050 / 2.0
        fg.dx = dx050  # 50m level
        fg.min_level_check = 4
        fg.tstart_max = tstart_max
        fg.tend_max = 1.e10  # when to stop monitoring max values
        fg.dt_check = 30.  # how often to update max values
        fg.interp_method = 0  # 0 ==> pw const in cells, recommended
        fgmax_grids.append(fg)  # written to fgmax_grids.data

        # # ###### Ishinomaki Bay
        # Points on a uniform 2d grid:
        fg = fgmax_tools.FGmaxGrid()
        fg.point_style = 2  # uniform rectangular x-y grid
        fg.x1 = 141.14 + dx050 / 2.0
        fg.x2 = 141.39 - dx050 / 2.0
        fg.y1 = 38.35 + dx050 / 2.0
        fg.y2 = 38.49 - dx050 / 2.0
        fg.dx = dx050  # 50m level
        fg.min_level_check = 4
        fg.tstart_max = tstart_max
        fg.tend_max = 1.e10  # when to stop monitoring max values
        fg.dt_check = 30.  # how often to update max values
        fg.interp_method = 0  # 0 ==> pw const in cells, recommended
        fgmax_grids.append(fg)  # written to fgmax_grids.data

        # ####### Rikuzentakata Bay
        # # Points on a uniform 2d grid:
        fg = fgmax_tools.FGmaxGrid()
        fg.point_style = 2  # uniform rectangular x-y grid
        fg.x1 = 141.59 + dx050 / 2.0
        fg.x2 = 141.69 - dx050 / 2.0
        fg.y1 = 38.96 + dx050 / 2.0
        fg.y2 = 39.04 - dx050 / 2.0
        fg.dx = dx050  # 50m level
        fg.min_level_check = 4
        fg.tstart_max = tstart_max
        fg.tend_max = 1.e10  # when to stop monitoring max values
        fg.dt_check = 30.  # how often to update max values
        fg.interp_method = 0  # 0 ==> pw const in cells, recommended
        fgmax_grids.append(fg)  # written to fgmax_grids.data

    return rundata
    # end of function setgeo
    # ----------------------

if __name__ == '__main__':
    # Set up run-time parameters and write all data files.
    import sys
    from clawpack.geoclaw import kmltools

    if making_B0:
        rundir = '_input/B0'
    else:
        rundir = '_input'

    os.makedirs(rundir, exist_ok=True)
    os.chdir(rundir)

    rundata = setrun(*sys.argv[1:])
    rundata.write()

    kmltools.make_input_data_kmls(rundata)
