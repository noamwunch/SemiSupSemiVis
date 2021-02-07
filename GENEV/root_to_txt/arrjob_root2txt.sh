#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N root2txt

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 1-50

root_macro_dir=$(pwd)
root_program_dir=/gpfs0/kats/projects/Delphes-3.4.2

root_file_dir="/gpfs0/kats/users/wunch/semisup_evs/xs_tt_GenMjjGt400_GenPtGt40_GenEtaSt3_MjjGt200_PtGt50_EtaSt2.5_new"
inp_file="$root_file_dir/$SGE_TASK_ID.root"

dijet="true"
veto_isolep="true"
PT_min=50
PT_max=100000
Eta_min=-2.5
Eta_max=2.5
Mjj_min=500
Mjj_max=100000
dR_jet=0.7

out_file=$inp_file.txt

source $HOME/.bash_profile
cd $root_program_dir
root -b << EOF
.x $root_macro_dir/root_tree_to_txt.C("$inp_file", $dijet, $veto_isolep, $PT_min, $PT_max, $Eta_min, $Eta_max, $Mjj_min, $Mjj_max, $dR_jet,"$out_file")
EOF

