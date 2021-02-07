#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N genevs

#$ -e ./stderr_bb_lv.txt
#$ -o ./stdout_bb_lv.txt

#$ -t 1-40

source $HOME/.bash_profile

. ./main_genev.sh $SGE_TASK_ID

