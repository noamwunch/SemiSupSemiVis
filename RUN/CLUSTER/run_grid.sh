### Run with:
### qsub -q kats.q -N <folder_name> -t <lines in folder_name/argseed.txt> array_job.sh
rm ./$1/std*
nlines=$(($(< "./iter_grid/argseed.txt" wc -l)-1))
qsub -q kats.q -N $1 -t 1-$nlines -l mem_free=50G -pe shared 10 array_job.sh
