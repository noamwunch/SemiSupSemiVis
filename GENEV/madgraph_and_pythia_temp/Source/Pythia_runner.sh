#!/bin/bash

# Transfer events to Pythia
cd $ScriptPath/CaseIandII/Events/$nameRun$i
gzip -d unweighted_events.lhe.gz
mv unweighted_events.lhe Hidden_valley_6.lhe
cp Hidden_valley_6.lhe $ScriptPath/examples

# Implement rinv
rvis=$(echo "x=1-$rinv; print 0; x" | bc)
cd $ScriptPath/Cards
cp main_hidden_valley_6.cmd $ScriptPath/examples/main_hidden_valley_6.cmd 
cd $ScriptPath/examples
sed -i -e "s/rinv/$rinv/g" main_hidden_valley_6.cmd
sed -i -e "s/rvis/$rvis/g" main_hidden_valley_6.cmd

# Introduce new parameters
sed -i -e "s/Removenfl/$nf025ns/g" main_hidden_valley_6.cmd
sed -i -e "s/RemovenGauge /$nGauge/g" main_hidden_valley_6.cmd
sed -i -e "s/RemovelambdaD/$lambdaD/g" main_hidden_valley_6.cmd
sed -i -e "s/RemoveDquarkMass/$DquarkMass/g" main_hidden_valley_6.cmd
sed -i -e "s/RemoveDpionMass/$DpionMass/g" main_hidden_valley_6.cmd
sed -i -e "s/RemoveDRhoMass/$DrhoMass/g" main_hidden_valley_6.cmd
sed -i -e "s/RemoveProbV/$ProbV/g" main_hidden_valley_6.cmd
sed -i -e "s/RemovedecayLength/$decayLength/g" main_hidden_valley_6.cmd

# Calculate pTmin for showering and replace it in code
pTminS=$(echo "x=1.1*$lambdaD; x" | bc)
sed -i -e "s/RemovepTmin/$pTminS/g" main_hidden_valley_6.cmd

cat main_hidden_valley_6.cmd

rm -f main_Hidden_valley_6.hepmc

# Run Pythia
./main_Hidden_valley_6


cd $ScriptPath
