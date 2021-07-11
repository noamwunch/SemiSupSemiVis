#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N genevs

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 1:2

source $HOME/.bash_profile

settings_file=$1

. ./main_genev.sh $SGE_TASK_ID "$settings_file"

