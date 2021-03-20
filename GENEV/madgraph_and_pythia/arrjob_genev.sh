#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N genevs

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 101:150

source $HOME/.bash_profile

. ./main_genev.sh $SGE_TASK_ID

