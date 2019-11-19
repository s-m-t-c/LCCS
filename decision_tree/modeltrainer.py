# Load modules
import concurrent.futures
import pickle
import sys
import warnings
import datacube
import fiona
import numpy as np
# import pandas as pd
import rasterio
import sklearn
import xarray as xr
import geopandas as gp
from datacube import helpers
from datacube.utils import geometry
from datacube.utils.geometry import CRS
from rasterio.features import rasterize
from sklearn import tree
# Import external functions from dea-notebooks using relative link to 10_Scripts
sys.path.append('/g/data/u46/users/sc0554/dea-notebooks/Scripts')
from dea_bandindices import calculate_indices
import dea_classificationtools


# 'Wrappers' to translate xarrays to np arrays and back for interfacing with sklearn models
def sklearn_flatten(input_xr):
    """
    Reshape a DataArray or Dataset with spatial (and optionally temporal) structure into
    an np.array with the spatial and temporal dimensions flattened into one dimension.

    This flattening procedure enables DataArrays and Datasets to be used to train and predict
    with sklearn models.

    Last modified: September 2019

    Parameters
    ----------
        input_xr : a DataArray or Dataset. Must have dimensions 'x' and 'y', may have dimension 'time'.
                   Dimensions other than 'x', 'y' and 'time' are unaffected by the flattening.

    Returns
    ----------
        input_np : a numpy array corresponding to input_xr.data (or input_xr.to_array().data), with
                   dimensions 'x','y' and 'time' flattened into a single dimension, which is the first
                   axis of the returned array. input_np contains no NaNs.

    """
    #     pdb.set_trace()
    # cast input Datasets to DataArray
    if isinstance(input_xr, xr.Dataset):
        input_xr = input_xr.to_array()

    # stack across pixel dimensions, handling timeseries if necessary
    if 'time' in input_xr.dims:
        stacked = input_xr.stack(z=['x', 'y', 'time'])
    else:
        stacked = input_xr.stack(z=['x', 'y'])

    # finding 'bands' dimensions in each pixel - these will not be flattened as their context is important for sklearn
    pxdims = []
    for dim in stacked.dims:
        if dim != 'z':
            pxdims.append(dim)

    # mask NaNs - we mask pixels with NaNs in *any* band, because sklearn cannot accept NaNs as input
    mask = np.isnan(stacked)
    if len(pxdims) != 0:
        mask = mask.any(dim=pxdims)

    # turn the mask into a numpy array (boolean indexing with xarrays acts weird)
    mask = mask.data
    # the dimension we are masking along ('z') needs to be the first dimension in the underlying np array for
    # the boolean indexing to work
    stacked = stacked.transpose('z', *pxdims)
    input_np = stacked.data[~mask]

    return input_np


# affine of single pixel is nans fyi
def get_training_data_for_shp(path, out_train, products=['ls8_nbart_tmad_annual'],
                              field='classnum'):
    """
    Function to extract data for training classifier

    Requires a list of products
    """
    data_list = []

    query = {'time': ('2015-01-01', '2015-12-31')}
    query['crs'] = 'EPSG:3577'
    shp = gp.read_file(path)
    bounds = shp.total_bounds
    minx = bounds[0]
    maxx = bounds[2]
    miny = bounds[1]
    maxy = bounds[3]
    query['x'] = (minx, maxx)
    query['y'] = (miny, maxy)

    # Make sure products is a list
    if not isinstance(products, list):
        products = [products]

    print("loading data...")
    for product in products:
        data = dc.load(product=product, group_by='solar_day', **query)
        # Check if geomedian is in the product and calculate indices if it is
        if "geomedian" in product:
            data = calculate_indices(data, 'BUI', collection='ga_ls_2')
            data = calculate_indices(data, 'BSI', collection='ga_ls_2')
            data = calculate_indices(data, 'BSI', collection='ga_ls_2')
            data = calculate_indices(data, 'NBI', collection='ga_ls_2')
            data = calculate_indices(data, 'EVI', collection='ga_ls_2')
            data = calculate_indices(data, 'NDWI', collection='ga_ls_2')
            data = calculate_indices(data, 'MSAVI', collection='ga_ls_2')
        # Remove time step if present

        try:
            data = data.isel(time=0)
        except ValueError:
            pass
            data_list.append(data)

    if len(products) == 1:
        data_all = data
    else:
        # FIXME: One for later
        raise Exception("Haven't implemented for multiple products yet")

    print("calculating indices...")
    # Calculate indices - will use for all features

    print("rastering features...")
    # Go through each feature
    i = 0
    for poly_geom, poly_class_id in zip(shp.geometry, shp[field]):
        print("Feature {:04}/{:04}\r".format(i + 1, len(shp.geometry)), end='')
        # Rasterise the feature
        mask = rasterize([(poly_geom, poly_class_id)],
                         out_shape=(data_all.y.size, data_all.x.size),
                         transform=data_all.affine)

        mask = xr.DataArray(mask, coords=(data_all.y, data_all.x))
        data_masked = data_all.where(mask == poly_class_id, np.nan)

        flat_train = sklearn_flatten(data_masked)
        flat_val = np.repeat(poly_class_id, flat_train.shape[0])
        #         print(flat_train.shape, flat_val.shape)
        stacked = np.hstack((np.expand_dims(flat_val, axis=1), flat_train))
        # Append training data and label to list
        out_train.append(stacked)
        i = i + 1

    # Return a list of labels for columns in output array
    return [field] + list(data_all.data_vars)

dc = datacube.Datacube(app = 'classifiers')

shp_list = glob.glob('/g/data1a/r78/LCCS_Aberystwyth/training_data/2015/*.shp')
out_train = []
for shp_num, path in enumerate(shp_list):
    print("[{:02}/{:02}]: {}".format(shp_num + 1, len(shp_list), path))
    try:
        column_names = get_training_data_for_shp(path, out_train, field='classnum')
    except Exception as e:
        print("Failed to extract data: {}".format(e))
    print("\n extracted pixels")

model_input = np.vstack(out_train)
print(model_input.shape)
np.savetxt("train_input_tmad.txt", model_input, header=' '.join(column_names), fmt='%.4f')