#!/bin/bash

#PBS -N fc_summary
#PBS -P u46
#PBS -q express
#PBS -l walltime=2:00:00
#PBS -l mem=128GB
#PBS -l jobfs=2GB
#PBS -l ncpus=1
#PBS -l wd
module use /g/data/v10/public/modules/modulefiles
module load dea
python /g/data/u46/users/sc0554/living_earth/fc_summary.py
