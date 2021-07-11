#!/bin/bash

###bb#####
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_bb.txt"
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_bb.sh"
# 1
cd madgraph_and_pythia
rm std*
qsub -N genevs_bb1 -t 1:50 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_bb1 -t 1:50 -hold_jid genevs_bb1 arrjob_root2txt.sh "$args_root"
cd ..

# 2
cd madgraph_and_pythia
rm std*
qsub -N genevs_bb2 -t 51:100 -hold_jid root2txt_bb1 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_bb2 -t 51:100 -hold_jid genevs_bb2 arrjob_root2txt.sh "$args_root"
cd ..

# 3
cd madgraph_and_pythia
rm std*
qsub -N genevs_bb3 -t 101:150 -hold_jid root2txt_bb2 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_bb3 -t 101:150 -hold_jid genevs_bb3 arrjob_root2txt.sh "$args_root"
cd ..

# 4
cd madgraph_and_pythia
rm std*
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_bb.txt"
qsub -N genevs_bb4 -t 151:200 -hold_jid root2txt_bb3 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_bb.sh"
qsub -N root2txt_bb4 -t 151:200 -hold_jid genevs_bb4 arrjob_root2txt.sh "$args_root"
cd ..

###cc#####
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_cc.txt"
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_cc.sh"
#1
cd madgraph_and_pythia
rm std*
qsub -N genevs_cc1 -t 1:50 -hold_jid root2txt_bb4 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_cc1 -t 1:50 -hold_jid genevs_cc1 arrjob_root2txt.sh "$args_root"
cd ..

#2
cd madgraph_and_pythia
rm std*
qsub -N genevs_cc2 -t 51:100 -hold_jid root2txt_cc1 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_cc2 -t 51:100 -hold_jid genevs_cc2 arrjob_root2txt.sh "$args_root"
cd ..

#3
cd madgraph_and_pythia
rm std*
qsub -N genevs_cc3 -t 101:150 -hold_jid root2txt_cc2 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_cc3 -t 101:150 -hold_jid genevs_cc3 arrjob_root2txt.sh "$args_root"
cd ..

#4
cd madgraph_and_pythia
rm std*
qsub -N genevs_cc4 -t 151:200 -hold_jid root2txt_cc3 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_cc4 -t 151:200 -hold_jid genevs_cc4 arrjob_root2txt.sh "$args_root"
cd ..

###dark_0.5mm_rinv0#####
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_dark_0.5mm_rinv0.txt"
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_dark_0.5mm_rinv0.sh"
cd madgraph_and_pythia
rm std*
qsub -N genevs_dark_0.5mm_rinv0 -t 1:50 -hold_jid root2txt_cc4 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_dark_0.5mm_rinv0 -t 1:50 -hold_jid genevs_dark_0.5mm_rinv0 arrjob_root2txt.sh "$args_root"
cd ..

###dark_0.25mm_rinv0#####
settings_madgraph="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/settings_madgraph/Settings_dark_0.25mm_rinv0.txt"
args_root="/gpfs0/kats/users/wunch/SemiSupSemiVis/GENEV/args_root/args_root_dark_0.25mm_rinv0.sh"
cd madgraph_and_pythia
rm std*
qsub -N genevs_dark_0.25mm_rinv0 -t 1:50 -hold_jid root2txt_dark_0.5mm_rinv0 arrjob_genev.sh "$settings_madgraph"
cd ..

cd root_to_txt
rm std*
qsub -N root2txt_dark_0.25mm_rinv0 -t 1:50 -hold_jid genevs_dark_0.25mm_rinv0 arrjob_root2txt.sh "$args_root"
cd ..