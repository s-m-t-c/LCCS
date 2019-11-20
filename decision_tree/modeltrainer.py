#!/usr/bin/env python
# Load modules
import argparse
import glob
import sys
import numpy as np

# Import external functions from dea-notebooks using relative link to 10_Scripts
sys.path.append('/g/data/u46/users/sc0554/dea-notebooks/Scripts')
sys.path.append('/home/552/dc4749/development/dea-notebooks/Scripts')
from dea_classificationtools import get_training_data_for_shp


def extract_data(product, output_file=None, feature_mean=False):
    shp_list = glob.glob('/g/data1a/r78/LCCS_Aberystwyth/training_data/2015/*.shp')
    out_train = []
    for shp_num, path in enumerate(shp_list):
        print("[{:02}/{:02}]: {}".format(shp_num + 1, len(shp_list), path))
        try:
            column_names = get_training_data_for_shp(path, out_train,
                                                     product=product,
                                                     time=('2015-01-01', '2015-12-31'),
                                                     crs='EPSG:3577', field='classnum',
                                                     calc_indices=False,
                                                     feature_mean=feature_mean)
        except Exception as e:
            print("Failed to extract data: {}".format(e))
        print("\n extracted pixels")

    model_input = np.vstack(out_train)
    np.savetxt("training_data_2015_{}.txt".format(product),
               model_input, header = ' '.join(column_names), fmt = '%.4f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training data")
    parser.add_argument("--product", type=str, required=False,
                        help="ODC product (e.g., ls8_nbart_geomedian_annual "
                             "or ls8_nbart_tmad_annual",
                        default="ls8_nbart_geomedian_annual")
    parser.add_argument("--mean", action='store_true', required=False,
                        help="Extract the mean of each feature rather than "
                              "each pixel",
                        default=False)
    args = parser.parse_args()

    extract_data(args.product, args.mean)
