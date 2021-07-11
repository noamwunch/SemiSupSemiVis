#!/bin/bash

#$ -cwd 
#$ -q kats.q@sge1050

#$ -N root2txt

#$ -e ./stderr.txt
#$ -o ./stdout.txt

#$ -t 1:2

args_script_path=$1

root_program_dir=/gpfs0/kats/projects/Delphes-3.4.2
root_macro_dir="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/root_to_txt"

. $args_script_path

inp_file="$root_inp_file_dir/$SGE_TASK_ID.root"
out_file="$txt_out_file_dir/$SGE_TASK_ID.root.txt"

mkdir -p $txt_out_file_dir
source $HOME/.bash_profile
cd $root_program_dir
root -b << EOF
.x $root_macro_dir/$root_macro.C("$inp_file", $dijet, $veto_isolep, $PT_min, $PT_max, $Eta_min, $Eta_max, $Mjj_min, $Mjj_max, $ystar_max, $dR_jet,"$out_file", $bkg_PID)
EOF

rm "$inp_file"

