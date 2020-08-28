from skimage.segmentation import quickshift, felzenszwalb
import xarray as xr
import scipy
import datacube
from skimage.measure import label, regionprops
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray, rio_slurp_reproject
from datacube.utils.geometry import GeoBox, box, CRS
from datacube.utils.cog import write_cog

# from sklearn.impute import SimpleImputer

#dc = datacube.Datacube(config='/home/547/sc0554/datacube.conf', env='lccs_dev')

#query = {'time': ('2015-01-01', '2015-12-31')}
#query['crs'] = 'EPSG:3577'

#data = dc.load(product='fc_percentile_albers_annual', measurements='PV_PC_90', **query)
data = xr.open_rasterio('/g/data/r78/LCCS_Aberystwyth/urban_tests/test_sites_peter/perth_2015_gm.tif')
data = assign_crs(data, crs='epsg:3577')
# quickshift expects multiband images with bands in the last dimension
data = data.transpose()
fname = '/g/data/r78/LCCS_Aberystwyth/continental_run_april2020/2015/lccs_2015_L4_0.5.0.tif'
LCCS = rio_slurp_xarray(fname, gbox=data.geobox)
LCCS = LCCS.isel(band=0)
print("LCCS shape", LCCS.shape)
meta_d = LCCS.copy()##.squeeze().drop('time')
seg = felzenszwalb(LCCS.data.transpose())
#seg = quickshift(LCCS.data.transpose(), kernel_size=3, convert2lab=False, max_dist=10, ratio=0.5)
print('seg shape', seg.shape)
data_seg_med = scipy.ndimage.median(input=LCCS.data.transpose(), labels=seg, index=seg)
#data_seg_med = data_seg_med.squeeze("time").drop("time")
print("seg_med shape", data_seg_med.shape)
out = xr.DataArray(data = data_seg_med.transpose(),dims = meta_d.dims, coords=meta_d.coords, attrs=meta_d.attrs)
print(out)
name = 'lccs_l4_2015_seg'
write_cog(out, fname=f'{name}.tif', overwrite=True)


# Create list of labels that do not meet the desired shape requirements
frac_dict = {}
# rect_dict = {}
# solidity_dict = {}
# form_dict = {}

# labels = []
#for region in regionprops(seg):
#    if region.area > 1:
#        fractal_dimension = 2 * np.log(region.perimeter / 4) / np.log(region.area)
        # rectangularity = region.area / (region.major_axis_length * region.minor_axis_length)
        # solidity = region.convex_area / region.area
        # form = (4 * np.pi * region.area) / np.square(region.perimeter)

        # Create dictionary of each property
#        frac_dict[region.label] = fractal_dimension
        # rect_dict[region.label] = rectangularity
        # solidity_dict[region.label] = solidity
        # form_dict[region.label] = form

        # Filter segments based on their region properties
        # if ((0.47 < rectangularity) & (rectangularity < 0.93)) & ((0.71 < solidity) & (solidity < 2.0)) & (
        #         (0.26 < form) & (form < 0.81)):
        #     labels.append(region.label)

# Create a mask using labels
# mask = np.isin(seg, labels)


# cult_area_fractal = np.ma.masked_array(tv_summary_seg, mask)

#frac_arr = np.vectorize(frac_dict.get)(seg).astype(float)

#meta_d = data.squeeze().drop('time')

#out = xr.Dataset({'cultfrac':(meta_d.dims,frac_arr)}, coords=meta_d.coords, attrs=meta_d.attrs)

#datacube.helpers.write_geotiff('natty_seg.tif', out)
