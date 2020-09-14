#!/usr/bin/env python
"""
Script to make NCI .pbs scripts to run extraction for 2015 shapefiles

Runs on the expressbw queue

After running this script can submit jobs using:

for jfile in `ls *pbs`; do qsub $PWD/$jfile; done`

"""

import os
import glob

# Output directory

out_dir = "/g/data/r78/LCCS_Aberystwyth/training_data/2015_extracted_geomedians"
# Get a list of shapefiles

shp_list = glob.glob("/g/data/r78/LCCS_Aberystwyth/training_data/2015/*shp")
# Run for multiple products

for product in ["ls8_nbart_geomedian_annual", "ls8_nbart_tmad_annual"]:
    for shp in shp_list:
        shp_basename = os.path.splitext(os.path.basename(shp))[0]
        out_jobs_script = os.path.join(out_dir, "{}_{}_job.pbs".format(shp_basename, product))
        out_geomedian_txt = os.path.join(out_dir, "{}_{}_stats.txt".format(shp_basename, product))
        out_mads_txt = os.path.join(out_dir, "{}_mads_stats.txt".format(shp_basename))

        out_job_text = """
#PBS -q expressbw
#PBS -P u46
#PBS -l mem=16GB
#PBS -l ncpus=1
#PBS -l walltime=02:00:00
#PBS -l wd

module use /g/data/v10/public/modules/modulefiles
module load dea
export PYTHONPATH=/home/547/sc0554/.digitalearthau/dea-env/20190709/local/lib/python3.6/site-packages:$PYTHONPATH

python {out_dir}/modeltrainer.py --mean --product {product} -o {out_geomedian_txt} {in_shp}

    """.format(in_shp=shp, out_geomedian_txt=out_geomedian_txt, out_dir = out_dir, out_mads_txt=out_mads_txt, product=product)

        with open(out_jobs_script, "w") as f:
            f.write(out_job_text)

        print("Wrote to: {}".format(out_jobs_script))
