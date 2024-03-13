"""
To preprocess elevation data files like gebco to create gauge points
Needs python-gis env in conda for geopandas and osgeo
"""
import os
import pandas as pd
import geopandas as gp
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import shapely.geometry
from shapely.ops import substring
import numpy as np

try:
    root_dir = os.environ['PTHA']
except:
    raise Exception("*** Must first set PTHA enviornment variable")

def get_contour_pts(filename, countour_depth, contour_interval, extent, makeplots=False):
    #reduce the extent to neccessary area
    temp_file = os.path.join(coastdir,'temp.tif')
    countour_line = str(countour_depth) + '_contour_line.shp'
    countour_point = str(countour_depth) + '_contour_point.shp'
    print('clipping data to extents:', extent)
    gdal.Translate(temp_file, filename, projWin=extent)

    # Read elevation raster with gdal,use first band,get the projection info and use as array
    elevraster = gdal.Open(temp_file)
    elevband = elevraster.GetRasterBand(1)
    elevproj = osr.SpatialReference(wkt=elevraster.GetProjection())
    elevArray = elevband.ReadAsArray()
    # print(Array[:4, :4])

    #set a nan value to use and check min/max
    elevNan = -32768
    elevMax = elevArray.max()
    elevMin = elevArray[elevArray!=elevNan].min()
    print("Maximum dem elevation: %.2f, minimum dem elevation: %.2f" % (elevMax, elevMin))

    #set info about contour line shapefile
    contourPath = os.path.join(coastdir, countour_line)
    contourDs = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(contourPath)
    contourShp = contourDs.CreateLayer('contour_line', elevproj)

    # define fields of id and elev
    fieldDef = ogr.FieldDefn("ID", ogr.OFTInteger)
    contourShp.CreateField(fieldDef)
    fieldDef = ogr.FieldDefn("elev", ogr.OFTReal)
    contourShp.CreateField(fieldDef)

    # Write shapefile using noDataValue
    # ContourGenerate(Band srcBand, double contourInterval, double contourBase, int fixedLevelCount, int useNoData,
    # double noDataValue,Layer dstLayer, int idField, int elevField
    gdal.ContourGenerate(elevband, 50.0, 1250.0, [countour_depth], 1, -32768., contourShp, 0, 1)
    contourDs.Destroy()
    os.remove(temp_file)
    #selecting main contour line
    gdf = gp.read_file(contourPath)
    gdf['length'] = gdf.to_crs(epsg=32654).geometry.length
    max_index = gdf['length'] .idxmax()
    print(" FID selected with max length:", max_index)
    gdf.to_file(contourPath)
    gdf[gdf['ID'] == max_index].to_file(contourPath)

    #splitting into equidistant points
    gdf = gp.read_file(contourPath)
    line = gdf.geometry[0]
    dis = contour_interval / (110 * 1000)
    mp = shapely.geometry.MultiPoint()

    for i in np.arange(0, line.length, dis):
        s = substring(line, i, i + dis)
        mp = mp.union(s.boundary)

    result = gp.GeoDataFrame(mp, columns=['geometry'])
    result.set_crs(epsg=4326)
    print('no of points generated are', result.shape[0])

    #Naming convention
    if countour_depth < - 50:
        Category = 'Deep' + str(countour_depth*-1)

    else:
        Category = 'Shallow' + str(countour_depth*-1)

    #Name,Type,Latitude,,Code
    result["ID"] = result.index + 1
    result["Name"] = Category
    result["Type"] = str(countour_depth*-1) + 'm'
    result["Latitude"] = round(result.centroid.y, 5)
    result["Longitude"] = round(result.centroid.x, 5)
    result["Code"] = result.index + (countour_depth * -1)*10000 + 1
    result["Name"] = result["Name"] + '_' + result["ID"].astype(str)
    pdresult = pd.DataFrame(result)
    pdresult = pdresult.drop(columns=['ID','geometry'])
    result.to_file(os.path.join(coastdir, countour_point))
    if os.path.exists(os.path.join(coastdir, 'prop_gauge_pts.csv')):
        pdresult.to_csv(os.path.join(coastdir, 'prop_gauge_pts.csv'), index=False, mode='a', header=False)
    else:
        pdresult.to_csv(os.path.join(coastdir, 'prop_gauge_pts.csv'), index=False, mode='a', header=True)


if __name__ == "__main__":
    print(os.getcwd())
    extent = [140.72349, 39.36857, 142.10877, 36.82250 ] #[ulx, uly, lrx, lry]
    Landfile = 'gis/topo/depth0050/COP30_JP0050m.tif'
    Waterfile = 'gis/topo/depth0150/JP0150m.tif'
    coastdir = 'gis/coastlines'

    #remove any old gauge file
    if os.path.exists(os.path.join(coastdir, 'prop_gauge_pts.csv')):
        os.remove(os.path.join(coastdir, 'prop_gauge_pts.csv'))

    #get_contour_pts(filename, countour_depth, contour_interval, extent, makeplots=False):

    # Deep points covered in water elevation file
    get_contour_pts(Waterfile, -100, 10*1000, extent)
    get_contour_pts(Waterfile, -50, 5*1000, extent)

    #Shallow points covered in land elevation file
    get_contour_pts(Landfile, -25, 2.5*1000, extent)
    get_contour_pts(Landfile, -5, 1*1000, extent)