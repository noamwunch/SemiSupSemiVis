#!/bin/bash

#$ -cwd

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -q kats.q

source $HOME/.bash_profile
conda activate ML

awk '{print $3}' argseed_sf0.025.txt | xargs python ../semisup.py

conda deactivate
