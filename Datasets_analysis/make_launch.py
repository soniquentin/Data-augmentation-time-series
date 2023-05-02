import sys
from params import *


if __name__ == "__main__" :


    nb_max_group = int(sys.argv[1])

    for i in range(1,nb_max_group+1) :
        print(f"launch{i}.sh : {DATASETS_TO_TEST[(i - 1)*group_size: i*group_size]}")

        script = f"""#!/bin/bash -l
#SBATCH -J MyGPUJob
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=quentin.lao@polytechnique.edu
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 7
#SBATCH --gpus=2
#SBATCH --time=47:59:00
#SBATCH -p gpu


conda activate tf-gpu
TF_CPP_MIN_LOG_LEVEL=2 python make_tests.py {i}
conda deactivate
            """

        with open(f"launch{i}.sh", 'w') as f :
            f.write(script)
    
