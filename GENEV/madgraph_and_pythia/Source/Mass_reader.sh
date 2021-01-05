#!/bin/bash

cd $ScriptPath

mZp=$(awk 'FNR==1 {print $1}' Grid_temp.dat)
mN=$(awk 'FNR==1 {print $2}' Grid_temp.dat)
rinv=$(awk 'FNR==1 {print $3}' Grid_temp.dat)

cd $ScriptPath
