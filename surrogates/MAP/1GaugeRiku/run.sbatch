#!/bin/bash
# options for sbatch
#SBATCH --job-name=trainTsuML-ishinomaki # Job name
#SBATCH --account=PR_carisma
#SBATCH --nodes=1 # number of nodes
#SBATCH --time=0-09:00 # time limit (D-HH:MM)
#SBATCH -p gpuq # partition
#SBATCH --nodelist=gp02
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --output ./run-%j.txt       # Standard out goes to this file
#SBATCH --error ./error-%j.txt        # Standard err goes to this file

# for calculating the amount of time the job takes
begin=`date +%s`
echo $HOSTNAME

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment
cd /mnt/beegfs/nragu/tsunami/japan/Paper/MAP/1GaugeRiku
# srun python process_fgmax.py

# setting up variables
zdim2use='20'
vae2use='100'
nepochs2use='20000'
epochs2use='0'

export nepochs2use
export zdim2use
export vae2use
export epochs2use 

 
for kfold in '0' '1' '2' '3' '4' 
do
    kfold2use=$kfold
    export kfold2use
        python vae_train.py 
        python vae_test.py
        python vae_test_historic.py
        python vae_plot.py
done

# finished commands
end=`date +%s`
elapsed=`expr $end - $begin`
echo ...
echo Time taken: $elapsed