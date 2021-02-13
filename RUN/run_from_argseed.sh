conda activate dark_jets

awk '{print $3}' argseed.txt | xargs python ../../../semisup.py