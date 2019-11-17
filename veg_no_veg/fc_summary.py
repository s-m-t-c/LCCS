import datacube
import matplotlib.pyplot as plt
from datacube import helpers
from datacube.utils import geometry
from scipy import ndimage
import xarray as xr
import numpy as np
import sys
sys.path.append('/g/data/u46/users/sc0554/dea-notebooks/10_Scripts/')
import DEADataHandling

dc=datacube.Datacube(config='/g/data/u46/users/sc0554/datacube.conf')

# x = (1500000, 1600000)
# y = (-4000000, -3900000)
x = (1199685, 1299651)
y = (-3800197, -3700025)
product = 'fc'
query = {'time': ('2015-01-01', '2015-12-31')}
query['x'] = (x[0], x[1])
query['y'] = (y[0], y[1])
query['crs'] = 'EPSG:3577'

mask_dict = {'cloud_acca': 'no_cloud',
             'cloud_fmask': 'no_cloud',
             'contiguous': True}

data = DEADataHandling.load_clearlandsat(dc=dc, query=query, product=product,
                                  masked_prop=0.1,
                                  mask_dict=mask_dict,
                                  satellite_metadata=True)
# Create a mask to show areas where total vegetation is greater than the bare-soil fraction of a pixel for
# each scene
tv_mask = data['BS'] < (data['PV'] + data['NPV'])
tv = tv_mask.where(data['PV'] > 0)
# Calculate the proportion of time where total vegetation is greater than the bare soil fraction of a pixel
# for the input year
tv_summary = tv.mean(dim='time', skipna=True)
# Create a boolean layer where vegetation is assigned if greater than .167
# tv_summary_filt = tv_summary > .167
# Convert booleans to binary
# tv_summary_filt = tv_summary_filt * 1
# Change data type to float
# tv_summary_filt = tv_summary_filt.data.astype(float)

meta_d = data.isel(time=0).drop('time')
out = xr.Dataset({'fc_summary': (meta_d.dims, tv_summary)}, coords=meta_d.coords, attrs=meta_d.attrs)
helpers.write_geotiff('testarea_pvfcsummary.tif', out)


tv_mask = data['BS'] < data['NPV']
tv = tv_mask.where(data['NPV'] > 0)
# Calculate the proportion of time where total vegetation is greater than the bare soil fraction of a pixel
# for the input year
tv_summary = tv.mean(dim='time')
# Create a boolean layer where vegetation is assigned if greater than .167
# tv_summary_filt = tv_summary > .167
# Convert booleans to binary
# tv_summary_filt = tv_summary_filt * 1
# Change data type to float
# tv_summary_filt = tv_summary_filt.data.astype(float)

meta_d = data.isel(time=0).drop('time')
out = xr.Dataset({'fc_summary': (meta_d.dims, tv_summary)}, coords=meta_d.coords, attrs=meta_d.attrs)
helpers.write_geotiff('cbr_npvfcsummary.tif', out)