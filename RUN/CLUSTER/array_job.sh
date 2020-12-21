#!/bin/bash

#$ -cwd

#$ -e ./$JOB_NAME/stderr.txt
#$ -o ./$JOB_NAME/stdout.txt

### Run with:
### qsub -q kats.q -N <folder_name> -t <lines in folder_name/argseed.txt> array_job.sh

source $HOME/.bash_profile
conda activate dark_jets

python ../../semisup.py `sed -n -e "$SGE_TASK_ID p" ./$JOB_NAME/argseed.txt`

conda deactivate
