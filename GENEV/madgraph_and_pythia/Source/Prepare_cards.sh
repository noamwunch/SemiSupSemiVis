#!/bin/bash

cd $ScriptPath/CaseIandII/Cards

# Put the run card in the card folder
cp $ScriptPath/Cards/run_card.dat run_card.dat

# Change the number of events to simulate
sed -i -e "s/ReplaceNEvents/$nEvents/g" run_card.dat

# Change the PDF
sed -i -e "s/ReplacePDF/$PDF/g" run_card.dat

# Change the beam energies
sed -i -e "s/ReplaceE1/$beamEnergy/g" run_card.dat
sed -i -e "s/ReplaceE2/$beamEnergy/g" run_card.dat
sed -i -e "s/ReplacepTMin/$pTMin/g" run_card.dat
sed -i -e "s/ReplacepTMax/$pTMax/g" run_card.dat
sed -i -e "s/ReplaceMjjMin/$MjjMin/g" run_card.dat
sed -i -e "s/ReplaceMjjMax/$MjjMax/g" run_card.dat

# Put the param_card in the card folder
cp $ScriptPath/Cards/param_card.dat param_card.dat

#Write mass information
sed -i -e "s/ReplacemZp/$mZp/g" param_card.dat
sed -i -e "s/ReplacemN/$mN/g" param_card.dat

cd $ScriptPath
