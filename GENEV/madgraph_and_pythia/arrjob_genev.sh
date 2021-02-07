#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N genevs

#$ -e ./stderr_tt_nvl.txt
#$ -o ./stdout_tt_nvl.txt

#$ -t 1-40

source $HOME/.bash_profile

. ./main_genev.sh $SGE_TASK_ID

