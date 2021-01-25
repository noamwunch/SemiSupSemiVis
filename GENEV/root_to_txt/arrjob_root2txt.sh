#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N root2txt

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 1-50

macro_dir=$(pwd)
root_program_dir=/gpfs0/kats/projects/Delphes-3.4.2

root_file_dir="/gpfs0/kats/users/wunch/semisup_evs/bkg_GenPtGt70"

inp_file="$root_file_dir/$SGE_TASK_ID.root"

dijet="true"
PT_min=100
PT_max=100000
dR_jet=0.7

out_file=$inp_file.txt

source $HOME/.bash_profile
cd $root_program_dir
root -b << EOF
.x $macro_dir/root_tree_to_txt.C("$inp_file",$dijet,$PT_min,$PT_max,$dR_jet,"$out_file")
EOF

