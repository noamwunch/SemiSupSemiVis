#!/bin/bash

#1) Create copy of Grid.dat and remove unecessary lines
tail -n +3 Grid.dat > Grid_temp.dat

#2) Read number of parameter space points to analyze
nPoints=$(wc -l < Grid_temp.dat)

#3) Set up MadGraph
cd $ScriptPath

#Remove any previous folder called "CaseIandII"
if [ -d "CaseIandII" ]; then
  rm -rf CaseIandII
fi

#Create the MG folder
/usr/bin/python2.7 $MGPath/bin/mg5_aMC <<EOF
set fortran_compiler /usr/bin/gfortran
import model $ScriptPath/models/${model}
${process[0]}
${process[1]}
${process[2]}
${process[3]}
${process[4]}
${process[5]}
${process[6]}
${process[7]}
${process[8]}
${process[9]}
output CaseIandII
exit
EOF

cd CaseIandII

#4) Remove the plot card
rm Cards/plot_card.dat

#5) Create result folder
mkdir -p $ResultPath
cd $ResultPath

#6) Begin summary file
touch Summary.txt
echo "####################" >> Summary.txt
echo "#   Summary file   #" >> Summary.txt
echo "####################" >> Summary.txt
echo "" >> Summary.txt
echo "Number of events" >> Summary.txt
echo $nEvents >> Summary.txt
echo "" >> Summary.txt
echo "Number of points in parameter space" >> Summary.txt
echo $nPoints >> Summary.txt
echo "" >> Summary.txt
echo "Model" >> Summary.txt
echo $model >> Summary.txt
echo "" >> Summary.txt
echo "PDF used" >> Summary.txt
echo $PDF >> Summary.txt
echo "" >> Summary.txt
echo "nGauge (The N of SU(N)))" >> Summary.txt
echo $nGauge >> Summary.txt
echo "" >> Summary.txt
echo "Number of dark quarks" >> Summary.txt
echo $nf025ns >> Summary.txt
echo "" >> Summary.txt
echo "Dark confinement scale in GeV" >> Summary.txt
echo $lambdaD >> Summary.txt
echo "" >> Summary.txt
echo "Dark quark mass in GeV" >> Summary.txt
echo $DquarkMass >> Summary.txt
echo "" >> Summary.txt
echo "Dark pion mass in GeV" >> Summary.txt
echo $DpionMass >> Summary.txt
echo "" >> Summary.txt
echo "Dark rho mass in GeV" >> Summary.txt
echo $DrhoMass >> Summary.txt
echo "" >> Summary.txt
echo "Probability to create dark rho" >> Summary.txt
echo $ProbV >> Summary.txt
echo "" >> Summary.txt
echo "Minimum pT of jet in GeV" >> Summary.txt
echo $pTMin >> Summary.txt
echo "" >> Summary.txt
echo "Maximum pT of jet in GeV" >> Summary.txt
echo $pTMax >> Summary.txt
echo "" >> Summary.txt
echo "Decay length of dark pions in mm" >> Summary.txt
echo $decayLength >> Summary.txt
echo "" >> Summary.txt
echo "MadGraph path" >> Summary.txt
echo $MGPath >> Summary.txt
echo "" >> Summary.txt
echo "Pythia path" >> Summary.txt
echo $PythiaPath >> Summary.txt
echo "" >> Summary.txt
echo "Delphes path" >> Summary.txt
echo $DelphesPath >> Summary.txt
echo "" >> Summary.txt

echo "Events kept" >> Summary.txt

if [ $KeepR == 1 ]; then
  echo "Yes" >> Summary.txt
else
  echo "No" >> Summary.txt
fi
echo "" >> Summary.txt

echo "Processes" >> Summary.txt

for ((i=0; i < 10; i++))
do
  if [ "${process[i]}" != "" ]; then
    echo ${process[i]} >> Summary.txt
  fi
done

echo "" >> Summary.txt
echo "Starting time" >> Summary.txt
echo $(date) >> Summary.txt

cd $ScriptPath
