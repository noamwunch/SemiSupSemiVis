### Run with:
### qsub -q kats.q -N <folder_name> -t <lines in folder_name/argseed.txt> array_job.sh

qsub -q kats.q -N example_grid -t 1-2 -l mem_free=50G array_job.sh
