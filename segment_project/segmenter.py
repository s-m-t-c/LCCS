from skimage.segmentation import quickshift
import xarray as xr
import scipy
import datacube
from skimage.measure import label, regionprops
# from sklearn.impute import SimpleImputer

dc = datacube.Datacube(config='/home/547/sc0554/datacube.conf', env='lccs_dev')

query = {'time': ('2015-01-01', '2015-12-31')}
query['crs'] = 'EPSG:3577'

data = dc.load(product='fc_percentile_albers_annual', measurements='PV_PC_90', **query)
seg = quickshift(data, kernel_size=7, convert2lab=False, max_dist=500, ratio=0.5)
data_seg_med = scipy.ndimage.median(input=data, labels=seg, index=seg)

# Create list of labels that do not meet the desired shape requirements
frac_dict = {}
# rect_dict = {}
# solidity_dict = {}
# form_dict = {}

# labels = []
for region in regionprops(seg):
    if region.area > 1:
        fractal_dimension = 2 * np.log(region.perimeter / 4) / np.log(region.area)
        # rectangularity = region.area / (region.major_axis_length * region.minor_axis_length)
        # solidity = region.convex_area / region.area
        # form = (4 * np.pi * region.area) / np.square(region.perimeter)

        # Create dictionary of each property
        frac_dict[region.label] = fractal_dimension
        # rect_dict[region.label] = rectangularity
        # solidity_dict[region.label] = solidity
        # form_dict[region.label] = form

        # Filter segments based on their region properties
        # if ((0.47 < rectangularity) & (rectangularity < 0.93)) & ((0.71 < solidity) & (solidity < 2.0)) & (
        #         (0.26 < form) & (form < 0.81)):
        #     labels.append(region.label)

# Create a mask using labels
# mask = np.isin(seg, labels)

# Mask the array
# cult_area_fractal = np.ma.masked_array(tv_summary_seg, mask)

frac_arr = np.vectorize(frac_dict.get)(seg).astype(float)

meta_d = data.squeeze().drop('time')

out = xr.Dataset({'cultfrac':(meta_d.dims,frac_arr)}, coords=meta_d.coords, attrs=meta_d.attrs)

datacube.helpers.write_geotiff('natty_seg.tif', out)