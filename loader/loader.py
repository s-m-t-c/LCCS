import datacube
import matplotlib.pyplot as plt
from datacube import helpers
from datacube.utils import geometry
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

meta_d = data.isel(time=0).drop('time')
out = xr.Dataset({'fc': (meta_d.dims, data)}, coords=meta_d.coords, attrs=meta_d.attrs)
helpers.write_geotiff('testarea.tif', out)