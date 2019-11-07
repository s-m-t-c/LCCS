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

dc = datacube.Datacube(app = 'classifiers')

# Open the shapefile
# shp_path='/g/data1a/u46/users/sc0554/LCCS/LCCS/decision_tree/training_data/training_samples_2015_-11_-35.shp'
shp_path='/g/data1a/u46/users/sc0554/LCCS/LCCS/decision_tree/training_data/LANDSCAPE_SALandCover_TrainingData_PointsConsolidated_SAOnly.shp'
shapes=fiona.open(shp_path,'r')
shp = gp.read_file(shp_path)
crs=geometry.CRS(shapes.crs_wkt)
# field = 'Classvalue'
field = 'CLASS_VALU'
class_value = 33
product = 'ls8_nbart_geomedian_annual'
query = {
         'time': ('2015-01-01', '2015-02-01')
         }
bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

# Filename to to save trained model
filename='dtree_model.sav'

def trainer(feature):
    # Datacube load the data near each feature in a shapefile, this returns a square / rectamg;e
    f_geometry=feature['geometry']
    geom=geometry.Geometry(f_geometry,crs=crs)
    query['geopolygon'] = geom
    data = dc.load(product=product, group_by='solar_day', **query)
    # Calculate indices
    data = calculate_indices(data, 'BUI', collection='ga_ls_2')
    data = calculate_indices(data, 'BSI', collection='ga_ls_2')
    data = calculate_indices(data, 'BSI', collection='ga_ls_2')
    data = calculate_indices(data, 'NBI', collection='ga_ls_2')
    data = calculate_indices(data, 'EVI', collection='ga_ls_2')
    data = calculate_indices(data, 'NDWI', collection='ga_ls_2')
    data = calculate_indices(data, 'MSAVI', collection='ga_ls_2')
    # Extract the label
    label = feature['properties'][field]
    # Append training data and label to list
    return (data, label)

with concurrent.futures.ProcessPoolExecutor() as executor:
    result = executor.map(trainer, shapes)

feature_rast_list = [x for x in result]

# Flatten arrays and append to list
flat_train_list = []
flat_val_list = []

for feature in feature_rast_list:
    # Flatten
    flat_train = sklearn_flatten(feature[0])
    # Make a list of labels for the same length as the training data
    flat_val = np.repeat(feature[1], flat_train.shape[0])
    flat_train_list.append(flat_train)
    flat_val_list.append(flat_val)

# Stack list off arrays into single array
val_input = np.hstack(flat_val_list)
train_input = np.vstack(flat_train_list)
print(train_input.shape, val_input.shape, )

# Initialise classifier
dtree = tree.DecisionTreeClassifier(random_state=0, max_depth=100)
# Fit classifier add "==x" to make a single class prediction.
dtree = dtree.fit(train_input, val_input)

pickle.dump(dtree, open(filename, 'wb'))
