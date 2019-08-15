from datacube.virtual import construct, Transformation, Measurement
from skimage.segmentation import quickshift
from scipy import stats
import numpy as np
from sklearn.impute import SimpleImputer

class Cultivated(Transformation):
    def compute(self, data):
        tmad = tmad.isel(time=0)
        tmad = tmad.drop('time')

        # Impute missing values (0 and NaN)
        imp_0 = SimpleImputer(missing_values=0, strategy='mean')
        container = {}
        for key in tmad.data_vars:
            d = tmad[key].data.squeeze()
            d = np.nan_to_num(d)
            d = np.where(d < 0, 0, d)
            d = np.where(d == 1, 0, d)
            imp_0.fit(d)
            d = imp_0.transform(d)
            d = -np.log(d)
            container.update({key: d})
        tmad['edev'].data = container['edev']
        tmad['sdev'].data = container['sdev']
        tmad['bcdev'].data = container['bcdev']

        # Calculate the mean of all the tmad inputs
        tmad_mean = np.mean(np.stack([tmad.edev.data, tmad.sdev.data, tmad.bcdev.data], axis=-1), axis=-1)
        # Convert type to float64 (required for quickshift)
        tmad_mean = np.float64(tmad_mean)
        # Segment
        tmad_seg = quickshift(tmad_mean, kernel_size=5, convert2lab=False, max_dist=500, ratio=0.5)
        # Calculate the median for each segment
        tmad_median_seg = scipy.ndimage.median(input=tmad_mean, labels=tmad_seg, index=tmad_seg)
        # Set threshold as 10th percentile of mean TMAD
        thresh = np.percentile(tmad_mean.ravel(), 10)
        # Create boolean layer using threshold
        tmad_thresh = tmad_median_seg < thresh
        # Convert from boolean to binary
        tmad_thresh = tmad_thresh * 1
        #tmad_thresh = tmad_thresh.astype(float)
        out = xr.Dataset({'cultman_agr_cat': (tmad.dims, tmad_thresh)}, coords=tmad.coords, attrs=tmad.attrs)
        return out
        #return tv_summary_filt.to_dataset(name='cultman_agr_cat')

    def measurements(self, input_measurements):
        return {'cultman_agr_cat': Measurement(name='cultman_agr_cat', dtype='float32', nodata=float('nan'), units='1')}

vegetat_veg_cat = construct(transform=Cultivated, input=dict(product='ls8_nbart_tmad_annual')

#vegetat_veg_cat_data = FC_summary.load(dc, **search_terms)

