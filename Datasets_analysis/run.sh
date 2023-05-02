python make_launch.py 15
for f in launch*;  do echo Running : sbatch ${f}; sbatch ${f}; rm ${f}; done;
