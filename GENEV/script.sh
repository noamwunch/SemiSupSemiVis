#!/bin/bash

###bb#####
cd madgraph_and_pythia
rm std*
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_bb.txt"
qsub -N genevs_cc -t 1:2 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_bb.sh"
qsub -N root2txt_cc -t 1:2 -hold_jid genevs_cc arrjob_root2txt.sh "$args_root"
cd ..

###cc#####
cd madgraph_and_pythia
rm std*
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_cc.txt"
qsub -N genevs_cc -t 1:2 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_cc.sh"
qsub -N root2txt_cc -t 1:2 -hold_jid genevs_cc arrjob_root2txt.sh "$args_root"
cd ..

###dark_0.5mm_rinv0#####
cd madgraph_and_pythia
rm std*
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_dark_0.5mm_rinv0.txt"
qsub -N genevs_cc -t 1:2 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_dark_0.5mm_rinv0.sh"
qsub -N root2txt_cc -t 1:2 -hold_jid genevs_cc arrjob_root2txt.sh "$args_root"
cd ..

###dark_0.25mm_rinv0#####
cd madgraph_and_pythia
rm std*
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_dark_0.25mm_rinv0.txt"
qsub -N genevs_cc -t 1:2 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_dark_0.25mm_rinv0.sh"
qsub -N root2txt_cc -t 1:2 -hold_jid genevs_cc arrjob_root2txt.sh "$args_root"
cd ..