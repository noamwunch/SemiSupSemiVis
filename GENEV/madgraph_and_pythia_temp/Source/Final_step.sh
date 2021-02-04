#!/bin/bash

# Remove temporary grid
rm Grid_temp.dat

# Write completion time
cd $ResultPath

echo "" >> Summary.txt
echo "Completion time" >> Summary.txt
echo $(date) >> Summary.txt

cd $ScriptPath
