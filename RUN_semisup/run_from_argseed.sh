conda activate ML
awk '{print $3}' argseed.txt | xargs python ../semisup.py
