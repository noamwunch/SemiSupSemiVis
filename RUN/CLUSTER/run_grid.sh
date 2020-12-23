### Run with:
### qsub -q kats.q -N <folder_name> -t <lines in folder_name/argseed.txt> array_job.sh
rm ./example_grid/std*
qsub -q kats.q -N example_grid -t 1-2 -l mem_free=50G -pe shared 10 array_job.sh
