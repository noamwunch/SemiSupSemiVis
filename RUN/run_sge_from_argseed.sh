#!/bin/bash

#$ -cwd

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -q kats.q

source $HOME/.bash_profile
conda activate ML

argseed="$1"

echo "$1"
echo "abcd"

awk '{print $3}' "$1" | xargs python ../semisup.py

conda deactivate
