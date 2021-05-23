#!/bin/bash

#$ -cwd

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -q kats.q@sge1050

source $HOME/.bash_profile
conda activate ML

awk '{print $3}' argseeds/argseed_lowmass_lessdata.txt | xargs python ../semisup_modular.py

conda deactivate
