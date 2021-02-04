#!/bin/bash

# Remove old events
if [ $KeepR != 1 ]; then
  cd $ScriptPath/CaseIandII
  ./bin/madevent <<EOF
  remove run_a$i
  y
  y
EOF
fi

# Remove events from Grid_temp.dat
cd $ScriptPath
sed -i '1d' Grid_temp.dat

cd $ScriptPath


