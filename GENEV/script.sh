#!/bin/bash

cd madgraph_and_pythia
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_cc.txt"
qsub -N genevs_cc -t 3:4 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_cc.txt"
qsub -N root2txt_cc -t 3:4 -hold_jid genevs_cc arrjob_genev.sh "$args_root"
cd ..