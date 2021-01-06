#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N test

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 1

source $HOME/.bash_profile

. ./main_gensig.sh $SGE_TASK_ID

