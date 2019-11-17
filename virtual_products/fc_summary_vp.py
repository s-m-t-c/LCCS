from datacube.virtual import Transformation, Measurement, construct_from_yaml, construct

class FC_summary(Transformation):
    def compute(self, data):
        # Create a mask to show areas where total vegetation is greater than the bare-soil fraction of a pixel for
        # each scene
        tv_mask = data['BS'] < (data['PV'] + data['NPV'])
        tv = tv_mask.where(data['PV'] > 0)
        # Calculate the proportion of time where total vegetation is greater than the bare soil fraction of a pixel
        # for the input year
        tv_summary = tv.nanmean(dim='time')
        # Create a boolean layer where vegetation is assigned if greater than .167
        tv_summary_filt = tv_summary > .167 # .5
        # Convert booleans to binary
        # tv_summary_filt = tv_summary_filt * 1
        tv_summary_filt = tv_summary_filt.data.astype('float')
        # Change data type to float
        # tv_summary_filt = tv_summary_filt.data.astype(float)

        out = xr.Dataset({'fc_summary': (meta_d.dims, tv_summary_filt)}, coords=meta_d.coords, attrs=meta_d.attrs)
        return out.to_dataset(name='vegetat_veg_cat')

    def measurements(self, input_measurements):
        return {'vegetat_veg_cat': Measurement(name='vegetat_veg_cat', dtype='float32', nodata=float('nan'), units='1')}

vsvg = construct(aggregate=FC_summary,
                 group_by='time',
                 input=dict(product='fc'))

fc_with_wofs = construct_from_yaml("""
    juxtapose:
      - product: ls8_fc_albers
      - product: wofs_albers
""")


vegetat_veg_cat = construct(transform=FC_summary,
                            input=fc_with_wofs)

vegetat_veg_cat_data = vegetat_veg_cat.load(dc, dask_chunks={'x': 512, 'y': 512, 'time': -1}, **search_terms)

