#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N genevs

#$ -e ./stderr_tt.txt
#$ -o ./stdout_tt.txt

#$ -t 21-50

source $HOME/.bash_profile

. ./main_genev.sh $SGE_TASK_ID

