#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N root2txt

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 61:100

root_macro_dir=$(pwd)
root_macro="root_tree_to_txt_with_rave_primvert"
root_program_dir=/gpfs0/kats/projects/Delphes-3.4.2

root_inp_file_dir="/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1"
txt_out_file_dir="/gpfs0/kats/users/wunch/semisup_dataset/sig_dl0.5_rinv0.00_mZp1500_lambda20_GenMjjGt800_GenPtGt40_GenEtaSt3_MjjGt1000_PtGt50_EtaSt2.5_y*lt1"
inp_file="$root_inp_file_dir/$SGE_TASK_ID.root"
out_file="$txt_out_file_dir/$SGE_TASK_ID.root.txt"

dijet="true"
veto_isolep="false"
PT_min=50
PT_max=100000
Eta_min=-2.5
Eta_max=2.5
Mjj_min=1000
Mjj_max=100000
dR_jet=0.7
ystar_min=1.0

mkdir -p $txt_out_file_dir
source $HOME/.bash_profile
cd $root_program_dir
root -b << EOF
.x $root_macro_dir/$root_macro.C("$inp_file", $dijet, $veto_isolep, $PT_min, $PT_max, $Eta_min, $Eta_max, $Mjj_min, $Mjj_max, $ystar_min, $dR_jet,"$out_file")
EOF

