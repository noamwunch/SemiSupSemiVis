#!/bin/bash

#$ -cwd

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -q kats.q

source $HOME/.bash_profile
conda activate ML

argseed=$1

awk '{print $3}' $argseed | xargs python ../semisup.py

conda deactivate
