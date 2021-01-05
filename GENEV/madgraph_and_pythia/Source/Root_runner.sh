#!/bin/bash

# Run root
cd $DelphesPath

root << EOF
.x Beauchesne/Example5D_track.C("results.root", $dRMax)
EOF

#Transfer files to result folder
mv Results.txt $ResultPath

#Rename file
cd $ResultPath
mv Results.txt run_a${i}.txt

cd $ScriptPath
