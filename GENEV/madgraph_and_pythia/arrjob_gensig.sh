#!/bin/bash

#$ -cwd

#$ -N

#$ -e ./stderr_$TASK_ID.txt
#$ -o ./stdout_$TASK_ID.txt

#$ -t

source $HOME/.bash_profile

. ./main_gensig.sh $SGE_TASK_ID

