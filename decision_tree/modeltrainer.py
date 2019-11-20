#!/usr/bin/env python3
"""
Script to extract training data from shape files to a text file.

Run using:

python3 modeltrainer.py --geomedian --product ls8_nbart_geomedian_annual ~/cultivated_classification/training_data/2015/*shp

"""

# Load modules
import argparse
import sys
import numpy as np

# Import external functions from dea-notebooks using relative link to 10_Scripts
# Sean's user on NCI
sys.path.append('/g/data/u46/users/sc0554/dea-notebooks/Scripts')
# Assume all repos are checked out to same location so get relative to this.
sys.path.append('../../dea-notebooks/Scripts')
from dea_classificationtools import get_training_data_for_shp


def extract_data(shp_list, product, output_file=None, feature_stats=None):
    out_train = []
    for shp_num, path in enumerate(shp_list):
        print("[{:02}/{:02}]: {}".format(shp_num + 1, len(shp_list), path))
        try:
            column_names = get_training_data_for_shp(path, out_train,
                                                     product=product,
                                                     time=('2015-01-01', '2015-12-31'),
                                                     crs='EPSG:3577', field='classnum',
                                                     calc_indices=False,
                                                     feature_stats=feature_stats)
        except Exception as e:
            print("Failed to extract data: {}".format(e))
        print("\n extracted pixels")

    model_input = np.vstack(out_train)
    if output_file is None:
        output_file = "training_data_2015_{}.txt".format(product)

    np.savetxt(output_file,
               model_input, header = ' '.join(column_names), fmt = '%.4f')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training data")
    parser.add_argument("inputshps", nargs="+",
                        help="Input shapefiles ")
    parser.add_argument("--product", type=str, required=False,
                        help="ODC product (e.g., ls8_nbart_geomedian_annual "
                             "or ls8_nbart_tmad_annual",
                        default="ls8_nbart_geomedian_annual")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mean", action='store_true', required=False,
                        help="Extract the mean of each feature rather than "
                              "each pixel",
                        default=False)
    group.add_argument("--geomedian", action='store_true', required=False,
                        help="Extract the geomedian of each feature rather than "
                              "each pixel",
                        default=False)
    args = parser.parse_args()

    feature_stats = None

    if args.mean:
        feature_stats = "mean"
    elif args.geomedian:
        feature_stats = "geomedian"

    extract_data(args.inputshps, args.product, feature_stats=feature_stats)



