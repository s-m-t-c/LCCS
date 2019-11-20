# Load modules
import glob
import sys
# Import external functions from dea-notebooks using relative link to 10_Scripts
sys.path.append('/g/data/u46/users/sc0554/dea-notebooks/Scripts')
from dea_classificationtools import get_training_data_for_shp

shp_list = glob.glob('/g/data1a/r78/LCCS_Aberystwyth/training_data/2015/*.shp')
out_train = []
for shp_num, path in enumerate(shp_list):
    print("[{:02}/{:02}]: {}".format(shp_num + 1, len(shp_list), path))
    try:
        column_names = get_training_data_for_shp(path, out_train,
                                                                         product='ls8_nbart_tmad_annual',
                                                                         time=('2015-01-01', '2015-12-31'),
                                                                         crs='EPSG:3577', field='classnum')
    except Exception as e:
        print("Failed to extract data: {}".format(e))
    print("\n extracted pixels")

model_input = np.vstack(out_train)
print(model_input.shape)
np.savetxt("train_input_tmad.txt", model_input, header = ' '.join(column_names), fmt = '%.4f')