#!/bin/bash

#Script location
ScriptPath=$(pwd)

#Output path
temp=$(sed -n -e '/^ResultPath/p' Settings.txt)
temp=$(echo $temp | xargs)
ResultPath=${temp#*=}

#MadGraph path
temp=$(sed -n -e '/^MGPath/p' Settings.txt)
temp=$(echo $temp | xargs)
MGPath=${temp#*=}

#Pythia path
temp=$(sed -n -e '/^PythiaPath/p' Settings.txt)
temp=$(echo $temp | xargs)
PythiaPath=${temp#*=}

#Delphes path
temp=$(sed -n -e '/^DelphesPath/p' Settings.txt)
temp=$(echo $temp | xargs)
DelphesPath=${temp#*=}

#Read number of events to generate
temp=$(sed -n -e '/^nEvents/p' Settings.txt)
temp=$(echo $temp | xargs)
nEvents=${temp#*=}

#Read model to load
temp=$(sed -n -e '/^model/p' Settings.txt)
temp=$(echo $temp | xargs)
model=${temp#*=}

#Read PDF
temp=$(sed -n -e '/^PDFchoice/p' Settings.txt)
temp=$(echo $temp | xargs)
PDF=${temp#*=}

#Read beam energy
temp=$(sed -n -e '/^beamEnergy/p' Settings.txt)
temp=$(echo $temp | xargs)
beamEnergy=${temp#*=}

#Read nGauge
temp=$(sed -n -e '/^nGauge/p' Settings.txt)
temp=$(echo $temp | xargs)
nGauge=${temp#*=}

#Read nf025ns
temp=$(sed -n -e '/^nf025ns/p' Settings.txt)
temp=$(echo $temp | xargs)
nf025ns=${temp#*=}

#Read dark confinement scale
temp=$(sed -n -e '/^lambdaD/p' Settings.txt)
temp=$(echo $temp | xargs)
lambdaD=${temp#*=}

#Read dark quark mass
temp=$(sed -n -e '/^DquarkMass/p' Settings.txt)
temp=$(echo $temp | xargs)
DquarkMass=${temp#*=}

#Read dark pion mass
temp=$(sed -n -e '/^DpionMass/p' Settings.txt)
temp=$(echo $temp | xargs)
DpionMass=${temp#*=}

#Read dark rho mass
temp=$(sed -n -e '/^DrhoMass/p' Settings.txt)
temp=$(echo $temp | xargs)
DrhoMass=${temp#*=}

#Read prbability to create vector meson
temp=$(sed -n -e '/^ProbV/p' Settings.txt)
temp=$(echo $temp | xargs)
ProbV=${temp#*=}

#Read whether to keep events or not
temp=$(sed -n -e '/^KeepR/p' Settings.txt)
temp=$(echo $temp | xargs)
KeepR=${temp#*=}

#Read minimum pT of jet
temp=$(sed -n -e '/^pTMin/p' Settings.txt)
temp=$(echo $temp | xargs)
pTMin=${temp#*=}

#Read maximum pT of jet
temp=$(sed -n -e '/^pTMax/p' Settings.txt)
temp=$(echo $temp | xargs)
pTMax=${temp#*=}

#Read minimum eta of jet
temp=$(sed -n -e '/^EtaMin/p' Settings.txt)
temp=$(echo $temp | xargs)
EtaMin=${temp#*=}

#Read maximum eta of jet
temp=$(sed -n -e '/^EtaMax/p' Settings.txt)
temp=$(echo $temp | xargs)
EtaMax=${temp#*=}

#Read minimum Mjj of event
temp=$(sed -n -e '/^MjjMin/p' Settings.txt)
temp=$(echo $temp | xargs)
MjjMin=${temp#*=}

#Read maximum Mjj of event
temp=$(sed -n -e '/^MjjMax/p' Settings.txt)
temp=$(echo $temp | xargs)
MjjMax=${temp#*=}

#Read radius around jet in which to keep tracks
temp=$(sed -n -e '/^dRMax/p' Settings.txt)
temp=$(echo $temp | xargs)
dRMax=${temp#*=}

#Read decay length of dark pions
temp=$(sed -n -e '/^decayLength/p' Settings.txt)
temp=$(echo $temp | xargs)
decayLength=${temp#*=}

#Read info processes
for ((i=0; i < 12; i++))
do
  process[i]=$(awk 'NR == n' n=$(($(grep -n "#List of processes" Settings.txt | cut -d : -f 1)+i+1)) Settings.txt)
  if [ $i = 0 ]; then
    if [ "${process[i]}" != "" ]; then
      process[i]="generate ${process[i]}"
    else
      echo "At least one process needed. Exiting..."
      exit
    fi
  else
    if [ "${process[i]}" != "" ]; then
      process[i]="add process ${process[i]}"
    else
      break
    fi
  fi
done

cd $ScriptPath
