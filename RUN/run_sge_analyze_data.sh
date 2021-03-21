#!/bin/bash

#$ -cwd

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -q kats.q@sge232

source $HOME/.bash_profile
conda activate ML

xargs python ../analyze_dataset.py

conda deactivate
