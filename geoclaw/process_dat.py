"""
To preprocess dat file from JP cabinet office and gebco
Needs python-gis env in conda for geopandas and osgeo
"""

import os
import pandas as pd
import geopandas as gp
from osgeo import gdal

root_dir = os.environ['PTHA']  # should point to main repo directory
print('root_dir = ', root_dir)

if __name__ == "__main__":
    TsunamiDir = root_dir

    tileinfofile = os.path.join(TsunamiDir, 'gis', 'topo', 'topotileinfo.csv')
    tileinfo = pd.read_csv(tileinfofile)
    exfile = gp.read_file('gis/topo/Extents.shp')

    NODATA_VALUE = -32768
    RES_VALUE = {"1350": 0.01215, "0450": 0.00405, "0150": 0.00135, "0050": 0.00045, "0030": 0.00027}


    # GEBCO_2021 main tile resample resolution
    filepath_gebco_in = 'gis/topo/gebco/gebco_2021_n50.0_s25.0_w130.0_e160.0.tif'
    filepath_gebco_out = 'gis/topo/ASC/depth_wgs_gebco.asc'
    print('Processing:', filepath_gebco_out)
    # gdal.Warp(filepath_gebco_out, filepath_gebco_in, dstSRS='EPSG:4326', xRes=RES_VALUE['0450'], yRes=RES_VALUE['0450'],
    #           srcNodata=-32768, dstNodata=-32768)

    # COP30 main tile resample resolution
    filepath_COP30_in = os.path.join(TsunamiDir, 'gis', 'topo', 'COP30','output_COP30.tif')
    filepath_COP30_out = 'gis/topo/ASC/depth_wgs_COP30.asc'
    print('Processing:', filepath_COP30_out)
    # gdal.Warp(filepath_COP30_out, filepath_COP30_in, dstSRS='EPSG:4326', xRes=RES_VALUE['0050'], yRes=RES_VALUE['0050'],
    #           srcNodata=0, dstNodata=-32768)

    # # JP DAT FILES
    for i in range(len(tileinfo[0:1])):
        #parameters
        tile_res = tileinfo.tile[i].split("-")[0]
        tile_id = tileinfo.tile[i].split("-")[1]
        print(tile_res + '-' + tile_id )

        #inputs
        filepath_in = 'gis/topo/depth' + tile_res + '/' + tile_res \
                      + 'm/depth_' + tileinfo.tile[i] + '.dat'
        filepath_ir = 'gis/topo/depth' + tile_res + '/ir' + tile_res \
                      + '/ir_' + tileinfo.tile[i] + '.dat'

        #intermediate outputs
        filepath_out = 'gis/topo/ASC/depth_' + tileinfo.tile[i] + '.asc'
        filepath_prjout = 'gis/topo/ASC/depth_' + tileinfo.tile[i] + '.prj'
        filepath_irout = 'gis/topo/ASC/ir_' + tileinfo.tile[i] + '.asc'
        filepath_irprjout = 'gis/topo/ASC/ir_' + tileinfo.tile[i] + '.prj'


        #final outputs
        filepath_wgsout = 'gis/topo/ASC/depth_wgs_' + tileinfo.tile[i] + '.asc'
        filepath_finalout = 'gis/topo/ASC/depth_COP30_' + tileinfo.tile[i] + '.asc'


        #Define ascii header info
        NCOLS = tileinfo.nx[i]
        NROWS = tileinfo.ny[i]
        XLLCORNER = tileinfo.xtl[i] - 300 #corrections for some shifts noticed
        YLLCORNER = tileinfo.ytl[i] - (tileinfo.ny[i]* tileinfo.dx[i]) + 350 #corrections for some shifts noticed
        CELLSIZE = tileinfo.dx[i]

        print(NCOLS,NROWS,XLLCORNER,YLLCORNER,CELLSIZE)

        prjline = 'PROJCS["unknown",GEOGCS["GCS_unknown",DATUM["D_Unknown_based_on_Bessel_1841_ellipsoid",' \
                  'SPHEROID["Bessel_1841",6377397.155,299.1528128]],PRIMEM["Greenwich",0.0],' \
                  'UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],' \
                  'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",143.0],PARAMETER["Scale_Factor",0.9996],' \
                  'PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]'

        with open(filepath_prjout, 'w') as f:
            f.write(prjline)

        topo = pd.read_fwf(filepath_in, header=None, widths=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8], delimiter="\n\t")

        lines = ['NCOLS ' + str(NCOLS), 'NROWS ' + str(NROWS), 'XLLCORNER ' + str(XLLCORNER),
                 'YLLCORNER ' + str(YLLCORNER),
                 'CELLSIZE ' + str(CELLSIZE), 'NODATA_VALUE ' + str(NODATA_VALUE)]
        with open(filepath_out, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')

        rtopo = pd.DataFrame(topo.values.reshape(NROWS, NCOLS)) * -1
        rtopo.round(decimals=3)
        rtopo.to_csv(filepath_out, sep=' ', header=False, index=None, mode='a')

        if tile_res == '0050':
            with open(filepath_irprjout, 'w') as f:
                f.write(prjline)

            topo = pd.read_fwf(filepath_ir, header=None, widths=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8], delimiter="\n\t")

            lines = ['NCOLS ' + str(NCOLS), 'NROWS ' + str(NROWS), 'XLLCORNER ' + str(XLLCORNER),
                     'YLLCORNER ' + str(YLLCORNER),
                     'CELLSIZE ' + str(CELLSIZE), 'NODATA_VALUE ' + str(NODATA_VALUE)]
            with open(filepath_irout, 'w') as f:
                for line in lines:
                    f.write(line)
                    f.write('\n')

            rtopo = ((pd.DataFrame(topo.values.reshape(NROWS, NCOLS))) % 1000 / 10)
            rtopo = rtopo.replace(0, NODATA_VALUE)
            rtopo.to_csv(filepath_irout, sep=' ', header=False, index=None, mode='a')


        # Find the extent to clip with from the shapefile
        raster = 'depth_wgs_' + tileinfo.tile[i] + '.asc'
        onetile = exfile.loc[exfile['Rastername'] == raster]
        corners = onetile.iloc[0].Extent[16:-3].split(',')
        xs = []
        ys = []

        for row in range(4): #iterate through xs and ys in multipoint info
            x = corners[row].split()[0]
            y = corners[row].split()[1]
            xs.append(float(x))
            ys.append(float(y))

        raster_ext = [round(min(xs), 2) + 0.005, round(min(ys), 2) + 0.005,
                      round(max(xs), 2) - 0.005, round(max(ys), 2) - 0.005]

        gdal.Warp(filepath_wgsout, filepath_out, dstSRS='EPSG:4326', xRes=RES_VALUE[tile_res],
                  yRes=RES_VALUE[tile_res], srcNodata=-32768, outputBounds=raster_ext)
        print('Processing:', filepath_wgsout, 'has extents:', raster_ext)

        if tile_res == '0050':
            gdal.Warp(filepath_finalout, [filepath_out, filepath_COP30_out, filepath_irout], dstSRS='EPSG:4326',
                      xRes=RES_VALUE[tile_res], yRes=RES_VALUE[tile_res], srcNodata=-32768, outputBounds=raster_ext)
            print('Processing:', filepath_out, 'has defense and extents:', raster_ext)



