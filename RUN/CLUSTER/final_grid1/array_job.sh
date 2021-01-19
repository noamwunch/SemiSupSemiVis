#!/bin/bash

#$ -cwd

#$ -e ./stderr.txt
#$ -o ./stdout.txt

source $HOME/.bash_profile
conda activate dark_jets

awk '{print $3}' argseed.txt | xargs python ../../../semisup.py

conda deactivate
