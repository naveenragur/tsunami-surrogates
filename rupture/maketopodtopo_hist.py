"""
Create topo and dtopo needed for this tohoku test:
Call functions with makeplots==True to create plots of topo, slip, and dtopo.
"""

from __future__ import absolute_import
from __future__ import print_function
from clawpack.geoclaw import dtopotools, topotools
from matplotlib import pyplot as plt
import math, pandas, os, numpy, glob

try:
    CLAW = os.environ['CLAW']
except:
    raise Exception("*** Must first set CLAW environment variable")

try:
    root_dir = os.environ['PTHA']
except:
    raise Exception("*** Must first set PTHA enviornment variable")


def Mw_2_M0(Mw, constant=9.05):
    """ Function that computes the Seismic Moment(M0) from Moment Magnitude(Mw). """
    M0 = 10 ** (Mw * 3 / 2 + constant)
    return M0

def M0_2_Mw(M0, constant=9.05):
    """ Function that computes the Seismic Moment(M0) from Moment Magnitude(Mw). """
    Mw = (2 / 3) * (math.log10(M0) - constant)
    return Mw

def Mw_2_RuptureDim(Mw):
#    Strasser, F. O., Arango, M. C., & Bommer, J. J. (2010).
    area_absigma = [-3.476, 0.952, 0.304]
    width_absigma = [-0.882, 0.351, 0.173]
    length_absigma = [-2.477, 0.585, 0.180]
    area = 10 ** (area_absigma[0] + Mw * area_absigma[1])
    width = 10 ** (width_absigma[0] + Mw * width_absigma[1])
    length = 10 ** (length_absigma[0] + Mw * length_absigma[1])
    output = [area, width, length]
    return output

def slip_from_Mw(Mw, mu=3.5e10, constant=9.05):
    M0 = Mw_2_M0(Mw, constant=constant)
    area = Mw_2_RuptureDim(Mw)[0] * 1e+06  # m^2
    slip = M0 / (area * mu)
    return slip

def make_dtopo(EveID, EveMw, EveLat, EveLon, EveDep, EveRak, EveStr ,EveDip, makeplots=False):
    GRID_SIZ = 0.01215
    X0, X1 = 135, 150
    Y0, Y1 = 30, 45

    XUNITS = round((X1 - X0) / GRID_SIZ)
    YUNITS = round((Y1 - Y0) / GRID_SIZ)

    # 'tohoku_dtopo_' + str(EveMw) + "_" + str(EveLat) + "_" + str(EveLon)
    # fullname replace with EveID

    dtopo_fname = os.path.join(DtopoDir,EveID + '.tt3')
    png_fname = os.path.join(PngDir, EveID + '.png')
    asc_fname = os.path.join(ASCDir, EveID + '.asc')
    prj_fname = os.path.join(ASCDir, EveID + 'prj')

    # For tohoku source defined by [Hayes (USGS, Tohoku 2011) ] , DIP= 10.21 degree, STRIKE = 194 degree and RAKE = 87.52 degree
    #create a subfault
    tohoku_fault = dtopotools.SubFault()
    # tohoku_fault.strike = 194
    # tohoku_fault.rake = 87.52
    # tohoku_fault.dip = 10.21
    tohoku_fault.strike = EveStr
    tohoku_fault.rake = EveRak
    tohoku_fault.dip = EveDip
    tohoku_fault.length = Mw_2_RuptureDim(EveMw)[2] * 1000
    tohoku_fault.width = Mw_2_RuptureDim(EveMw)[1] * 1000
    tohoku_fault.depth = EveDep * 1000
    tohoku_fault.slip = slip_from_Mw(EveMw)
    tohoku_fault.longitude = EveLon
    tohoku_fault.latitude = EveLat
    tohoku_fault.coordinate_specification = "top center"
    # create a fault and add the above single subfault to it
    fault = dtopotools.Fault()
    fault.subfaults = [tohoku_fault]

    x = numpy.linspace(X0, X1, XUNITS)
    y = numpy.linspace(Y0, Y1, YUNITS)
    times = [1.]

    fault.create_dtopography(x, y, times)
    dtopo = fault.dtopo
    dtopo.write(dtopo_fname, dtopo_type=3)
    print("Created ", dtopo_fname)

    tt3_file = open(dtopo_fname, "r")
    lines = tt3_file.readlines()
    tt3_file.close()
    asc_file = open(asc_fname, "w")
    asc_file.write('ncols ' + str.split(lines[0])[0] + '\n')
    asc_file.write('nrows ' + str.split(lines[1])[0] + '\n')
    asc_file.write('xllcorner ' + str.split(lines[3])[0] + '\n')
    asc_file.write('yllcorner ' + str.split(lines[4])[0] + '\n')
    asc_file.write('cellsize ' + str(GRID_SIZ) + '\n')
    asc_file.write('nodata_value -32768' + '\n')

    for line in lines[9:]:
        asc_file.write(line)
    asc_file.close()
    print("Created ", asc_fname)

    prjline = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],' \
              'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
    with open(prj_fname, 'w') as f:
        f.write(prjline)

    if makeplots:
        shore = numpy.load(shorelines_file)
        plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        fault.plot_subfaults(axes=ax1, slip_color=True)
        ax1.plot(EveLon, EveLat, 'ro')
        ax1.text(142, 32, ('ID:' + str(EveID)), fontsize=11)
        ax1.text(142, 31.2, ('Loc:' + str("{:.3f}".format(EveLat)) + ',' + str("{:.3f}".format(EveLon))), fontsize=11)
        ax1.text(142, 30.4, ('Mw,Depth :' + str(EveMw) + ',' + str(EveDep) + 'km'), fontsize=11)
        ax1.plot(shore[:, 0], shore[:, 1], 'g')
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(y.min(), y.max())
        dtopo.plot_dZ_colors(1., axes=ax2)
        ax2.plot(EveLon, EveLat, 'ro')
        ax2.text(142, 32, ('Dip:' + str(EveDip) + ' degree'), fontsize=11)
        ax2.text(142, 31.2, ('Rak:' + str(EveRak) + ' degree'), fontsize=11)
        ax2.text(142, 30.4, ('Str:' + str(EveStr) + ' degree'), fontsize=11)
        ax2.plot(shore[:, 0], shore[:, 1], 'g')
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(y.min(), y.max())
        plt.savefig(png_fname)
        plt.clf()
        plt.close('all')
        print("Created ", png_fname)


def make_hisdtopo(tohoku_subfaults, fname, makeplots=False,makeasc=True):
    dtopo_fname = os.path.join(DtopoDir, fname + '.tt3')
    png_fname = os.path.join(PngDir, fname + '.png')
    asc_fname = os.path.join(ASCDir, fname + '.asc')
    prj_fname = os.path.join(ASCDir, fname + '.prj')

    GRID_SIZ = 0.01215
    X0, X1 = 139, 145
    Y0, Y1 = 34, 42
    X0, X1 = 132, 145
    Y0, Y1 = 32, 42

    XUNITS = round((X1 - X0) / GRID_SIZ)
    YUNITS = round((Y1 - Y0) / GRID_SIZ)

    x = numpy.linspace(X0, X1, XUNITS)
    y = numpy.linspace(Y0, Y1, YUNITS)

    if tohoku_subfaults.shape[1] < 13:
        print('type 1')
        times = [1.]
        fault = dtopotools.Fault()
        fault.subfaults = []
        for i in range(len(tohoku_subfaults)):
            subfault = dtopotools.SubFault()
            subfault.length = tohoku_subfaults.L[i] * 1000
            subfault.width = tohoku_subfaults.W[i] * 1000
            subfault.dip = tohoku_subfaults.DIP[i]
            subfault.strike = tohoku_subfaults.STRIKE[i]
            subfault.depth = tohoku_subfaults.D[i] * 1000
            subfault.slip = tohoku_subfaults.SLIP[i]
            subfault.rake = tohoku_subfaults.RAKE[i]
            subfault.longitude = tohoku_subfaults.LON[i]
            subfault.latitude = tohoku_subfaults.LAT[i]
            subfault.coordinate_specification = 'top center'
            fault.subfaults.append(subfault)
        fault.rupture_type = 'static'
        fault.create_dtopography(x, y, times=times)
        dtopo = fault.dtopo
        dtopo.write(dtopo_fname, dtopo_type=3)
        print("Created ", dtopo_fname)

    elif tohoku_subfaults.shape[1] == 13:
        times = numpy.linspace(0, 160, 20)
        fault = dtopotools.Fault()
        fault.subfaults = []
        for i in range(len(tohoku_subfaults)):
            subfault = dtopotools.SubFault()
            subfault.length = tohoku_subfaults.L[i] * 1000
            subfault.width = tohoku_subfaults.W[i] * 1000
            subfault.dip = tohoku_subfaults.DIP[i]
            subfault.strike = tohoku_subfaults.STRIKE[i]
            subfault.depth = tohoku_subfaults.D[i] * 1000
            subfault.slip = tohoku_subfaults.SLIP[i]
            subfault.rake = tohoku_subfaults.RAKE[i]
            subfault.longitude = tohoku_subfaults.LON[i]
            subfault.latitude = tohoku_subfaults.LAT[i]
            subfault.rupture_time = tohoku_subfaults.Tinit[i]
            subfault.rise_time = tohoku_subfaults.Trise[i]
            subfault.coordinate_specification = 'bottom center'
            fault.subfaults.append(subfault)
        fault.rupture_type = 'dynamic'
        fault.create_dtopography(x, y, times=times)
        dtopo = fault.dtopo
        dtopo.write(dtopo_fname, dtopo_type=3)
        print("Created ", dtopo_fname)

    elif tohoku_subfaults.shape[1] > 13:
        makeasc=False
        makeplots=False
        times = [1.]
        fault = dtopotools.Fault()
        fault.subfaults = []
        for j in range(12, 23):
            print(str((j-12)*0.5) + 'min slip')
            for i in range(len(tohoku_subfaults)):
                subfault = dtopotools.SubFault()
                subfault.length = tohoku_subfaults.L[i] * 1000
                subfault.width = tohoku_subfaults.W[i] * 1000
                subfault.dip = tohoku_subfaults.DIP[i]
                subfault.strike = tohoku_subfaults.STRIKE[i]
                subfault.depth = tohoku_subfaults.D[i] * 1000
                subfault.slip = tohoku_subfaults.iloc[i, j]
                subfault.rake = tohoku_subfaults.RAKE[i]
                subfault.longitude = tohoku_subfaults.LON[i]
                subfault.latitude = tohoku_subfaults.LAT[i]
                subfault.coordinate_specification = 'top center'
                fault.subfaults.append(subfault)
            fault.rupture_type = 'dynamic'
            fault.create_dtopography(x, y, times=times)
            dtopo = fault.dtopo
            temp_fname = os.path.join(DtopoDir, fname +'_'+ str((j-12)*0.5) + 'min' + '.tt3')
            dtopo.write(temp_fname, dtopo_type=3)
            print("Created temp file", temp_fname)
            temp = open(temp_fname, "r")
            lines = temp.readlines()
            temp.close()
            if j == 12:
                tt3_file = open(dtopo_fname, 'w')
                tt3_file.write('    ' + str.split(lines[0])[0] + '       mx ' + '\n')
                tt3_file.write('    ' + str.split(lines[1])[0] + '       my ' + '\n')
                tt3_file.write('    ' + str(len(range(12, 23))) + '       mt ' + '\n')
                tt3_file.write(str.split(lines[3])[0] + '   xlower ' + '\n')
                tt3_file.write(str.split(lines[4])[0] + '   ylower ' + '\n')
                tt3_file.write('0.00000000000000e+00   t0 ' + '\n')
                tt3_file.write(str.split(lines[6])[0] + '   dx ' + '\n')
                tt3_file.write(str.split(lines[7])[0] + '   dy ' + '\n')
                tt3_file.write('3.0000000000000e+01   dt ' + '\n')
                for line in lines[9:]:
                    tt3_file.write(line)
                tt3_file.close()
            else:
                tt3_file = open(dtopo_fname, 'a')
                for line in lines[9:]:
                    tt3_file.write(line)
                tt3_file.close()
            os.remove(temp_fname)
        print("Created file", dtopo_fname)

    if makeasc:
        tt3_file = open(dtopo_fname, "r")
        lines = tt3_file.readlines()
        tt3_file.close()
        asc_file = open(asc_fname, "w")
        asc_file.write('ncols ' + str.split(lines[0])[0] + '\n')
        asc_file.write('nrows ' + str.split(lines[1])[0] + '\n')
        asc_file.write('xllcorner ' + str.split(lines[3])[0] + '\n')
        asc_file.write('yllcorner ' + str.split(lines[4])[0] + '\n')
        asc_file.write('cellsize ' + str(GRID_SIZ) + '\n')
        asc_file.write('nodata_value -32768' + '\n')

        for line in lines[9:]:
            asc_file.write(line)
        asc_file.close()
        print("Created ", asc_fname)

        prjline = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],' \
                  'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
        with open(prj_fname, 'w') as f:
            f.write(prjline)

    if makeplots:
        shore = numpy.load(shorelines_file)
        plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        if fault.rupture_type == 'dynamic':
            fault.plot_subfaults(axes=ax1, slip_color=True, plot_box=False, slip_time=max(times))
        else:
            fault.plot_subfaults(axes=ax1, slip_color=True, plot_box=False)
        ax1.plot(shore[:, 0], shore[:, 1], 'g')
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(y.min(), y.max())
        dtopo.plot_dZ_colors(max(times), axes=ax2)
        ax2.plot(shore[:, 0], shore[:, 1], 'g')
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(y.min(), y.max())
        plt.savefig(png_fname)
        #plt.clf()
        #plt.close('all')
        print("Created ", png_fname)


def get_topo(makeplots=False):
    """
    Retrieve the topo file and plot png.
    """
    if 1:
        print(os.getcwd())
        topo = topotools.Topography(path=root_dir + '/gis/topo/ASC/depth_wgs_gebco.asc', topo_type=2)
        shore = topo.make_shoreline_xy()
        plt.plot(shore[:, 0], shore[:, 1])
        plt.savefig(root_dir + '/gis/coastlines/gebco_coastline.png')
        numpy.save(shorelines_file, shore)

    if makeplots:
        topo_filepaths = glob.glob(root_dir + '/gis/topo/ASC/*wgs*.asc')
        for topofile in topo_filepaths:
            topo = topotools.Topography(topofile, topo_type=2)
            topo.plot()
            fname = os.path.splitext(topofile)[0] + '.png'
            plt.savefig(fname)
            plt.clf()
            plt.close('all')
            print("Created plot for:", fname)


if __name__ == "__main__":
    # Directory for storing topo and dtopo files:
    DtopoDir = os.path.join(root_dir, 'gis', 'dtopo_his')
    TopoDir = os.path.join(root_dir, 'gis', 'topo', 'ASC')
    PngDir = os.path.join(DtopoDir, 'dtopopng')
    ASCDir = os.path.join(DtopoDir, 'dtopoasc')
    os.makedirs(DtopoDir, exist_ok=True)
    os.makedirs(PngDir, exist_ok=True)
    os.makedirs(ASCDir, exist_ok=True)
    shorelines_file = root_dir + '/rupture/_inputs/JPshoreline_xy.npy'

########################### DTOPO FILES ################################
# -----------------------------------------------------------------------------------------------------------------------
    # # 2011 historic event in Tohoku Region v2
    # fnames = glob.glob(root_dir + '/gis/2011/source/_input2script/*.csv*')
    # fnames.remove(root_dir + '/gis/2011/source/_input2script/SATAKE2013.csv')

    fnames = ['SatakeMiniUpper','SatakeMiniUpperSoft','SatakeMiniLower','SatakeMiniLowerSoft']
    for count, name in enumerate(fnames):
        fname = root_dir + '/rupture/_inputs/historic/' + name +'.csv'
        subfaultdf = pandas.read_csv(fname)
        print('processing file',count,fname.split('/')[-1][:-4])
        make_hisdtopo(subfaultdf, fname.split('/')[-1][:-4], True, True)

# -----------------------------------------------------------------------------------------------------------------------
#      # # Sythetic Events in Tohoku Region based on TMOD from YAMAZAKI 2018
#     subfaultdf = pandas.read_csv(os.path.join(root_dir, 'gis', '2011', 'source', '_input2script', 'YAMAZAKI2018_TMOD.csv'))
#     MW_VALS = [7.5, 8, 8.5, 9]
#     COMPILE = pandas.DataFrame(columns=['EveID', 'EveMw', 'EveLat', 'EveLon', 'EveDep', 'EveRak', 'EveStr', 'EveDip'])
#     id = 1
#     for EveMw in MW_VALS:
#         for i in range(len(subfaultdf)):
#             #generate event ids
#             EveID = str('SC_' + format(id, '04d'))
#             id += 1
#
#             # create a compilation with event details
#             COMPILE = COMPILE.append({'EveID': EveID, 'EveMw': EveMw,'EveLat': subfaultdf.LAT[i],
#                                       'EveLon': subfaultdf.LON[i], 'EveDep': subfaultdf.D[i],
#                                       'EveRak': subfaultdf.RAKE[i], 'EveStr': subfaultdf.STRIKE[i],
#                                       'EveDip': subfaultdf.DIP[i]}, ignore_index=True)
#
#             #create dtopo files
#             make_dtopo(EveID, EveMw, subfaultdf.LAT[i], subfaultdf.LON[i], subfaultdf.D[i], subfaultdf.RAKE[i],
#                        subfaultdf.STRIKE[i], subfaultdf.DIP[i], makeplots=True)
#
#     COMPILE.to_csv(os.path.join(DtopoDir, 'tohoku_dtopo_list.csv'), sep=',', header=True, index=None, mode='w')

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
    #  # # Sythetic Events in Tohoku Region based on SLAB.2 data and custom source region with points at .25 degree spacing
    # subfaultdf = pandas.read_csv(os.path.join(root_dir, 'gis', 'GEM', 'sourceparameters.csv'))
    # MW_VALS = [7.5, 8, 8.5, 9, 9.5]
    # COMPILE = pandas.DataFrame(columns=['EveID', 'EveMw', 'EveLat', 'EveLon', 'EveDep', 'EveRak', 'EveStr', 'EveDip'])
    # id = 0
    # for EveMw in MW_VALS:
    #     for i in range(len(subfaultdf)):
    #         #generate event ids
    #         EveID = str('SL_' + format(id, '04d'))
    #         id += 1

    #         # create a compilation with event details
    #         COMPILE = COMPILE.append({'EveID': EveID, 'EveMw': EveMw,'EveLat': subfaultdf.lat[i],
    #                                   'EveLon': subfaultdf.lon[i], 'EveDep': abs(subfaultdf.dep[i]),
    #                                   'EveRak': subfaultdf.rak[i], 'EveStr': subfaultdf.str[i],
    #                                   'EveDip': subfaultdf.dip[i]}, ignore_index=True)

    #         #create dtopo files
    #         make_dtopo(EveID, EveMw, subfaultdf.lat[i], subfaultdf.lon[i], abs(subfaultdf.dep[i]), subfaultdf.rak[i],
    #                    subfaultdf.str[i], subfaultdf.dip[i], makeplots=True)

    # COMPILE.to_csv(os.path.join(DtopoDir, 'slab_dtopo_list.csv'), sep=',', header=True, index=None, mode='w')

# -----------------------------------------------------------------------------------------------------------------------
#     TOPO FILES
#     get_topo(True)
