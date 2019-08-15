#!/bin/bash

#PBS -N segmenter
#PBS -P u46
#PBS -q megamem
#PBS -l walltime=24:00:00
#PBS -l mem=3TB
#PBS -l jobfs=2GB
#PBS -l ncpus=32
#PBS -l wd
module use /g/data/v10/public/modules/modulefiles
module load dea
python /g/data/u46/users/sc0554/living_earth/cult_area/segmenter.py
