#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N root2txt

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 1-2

root_macro_dir=$(pwd)
root_program_dir=/gpfs0/kats/projects/Delphes-3.4.2

root_file_dir="/gpfs0/kats/users/wunch/semisup_evs/sig_rinv0.25_mZp1000_GenMjjGt400_GenPtGt40_GenEtaSt3_MjjGt500_PtGt50_EtaSt2.5/train"

inp_file="$root_file_dir/$SGE_TASK_ID.root"

dijet="true"
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
.x $root_macro_dir/root_tree_to_txt.C("$inp_file", $dijet, $PT_min, $PT_max, $Eta_min, $Eta_max, $Mjj_min, $Mjj_max $dR_jet,"$out_file")
EOF

