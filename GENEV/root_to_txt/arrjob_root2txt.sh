#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N test

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 1

root_program_dir=/gpfs0/kats/projects/Delphes-3.4.2

root_file_dir="/gpfs0/kats/users/wunch/semisup_evs"
mZp=500
rinv=0

inp_file="$root_file_dir/sig_$SGE_TASK_ID.$mZp.$rinv.root"

dijet="true"
PT_min=50
PT_max=10000
dR_jet=0.7

out_file=$inp_file.txt

source $HOME/.bash_profile
cd $root_program_dir
root -b << EOF
.x ./root_tree_to_txt.C("inp_file",
                        $dijet,
                        $PT_min,
                        $PT_max,
                        "$out_file")
EOF

