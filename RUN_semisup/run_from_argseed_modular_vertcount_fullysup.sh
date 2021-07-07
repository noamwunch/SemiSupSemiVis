conda activate ML
awk '{print $3}' ./argseeds/argseed.txt | xargs python ../semisup_modular_vertcount_fullysup.py
