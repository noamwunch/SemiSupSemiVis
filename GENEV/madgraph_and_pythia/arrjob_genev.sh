#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N genevs

#$ -e ./stderr_tt_new.txt
#$ -o ./stdout_tt_new.txt

#$ -t 1-50

source $HOME/.bash_profile

. ./main_genev.sh $SGE_TASK_ID

