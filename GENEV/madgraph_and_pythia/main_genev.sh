#!/bin/bash

### Steup ###
TaskID=$1
settings_file=$2
source ~/.bash_profile
cp -r ../madgraph_and_pythia $TMPDIR/madgraph_and_pythia
cd $TMPDIR/madgraph_and_pythia
#############

#1) Read settings
rm Settings.txt
echo "$settings_file"
cp "$settings_file" Settings.txt
cat Settings.txt
. ./Source/Setting_reader.sh

#2) Preliminary steps
. ./Source/Preliminary_steps.sh

#3) Start loop
for ((i=0; i < nPoints; i++))
do

#4) Mass reader
. ./Source/Mass_reader.sh

#5) Prepare cards
. ./Source/Prepare_cards.sh

#6) Run MadGraph
cd $ScriptPath/CaseIandII
/usr/bin/python2.7 $ScriptPath/CaseIandII/bin/madevent <<EOF
set fortran_compiler /usr/bin/gfortran
launch run_a$i
EOF
cd $ScriptPath

nameRun="run_a"

#7) Run Pythia
. ./Source/Pythia_runner.sh

#8) Delphes
OutName="$TaskID.root"
. ./Source/Delphes_applier.sh

#10) End loop
. ./Source/End_loop.sh
done

#11) Concluding step
. ./Source/Final_step.sh
