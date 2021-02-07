#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N genevs

#$ -e ./stderr_bb.txt
#$ -o ./stdout_bb.txt

#$ -t 21-40

source $HOME/.bash_profile

. ./main_genev.sh $SGE_TASK_ID

