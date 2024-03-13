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
cd /mnt/beegfs/nragu/tsunami/japan/Paper/TS/_ishi_5675

# setting up variables
zdim2use='150'
vae2use='100'
nepochs2use='3000'

export nepochs2use
export zdim2use
export vae2use

for epochs2use in '3000'
do 
    epochs2use=$epochs2use
    export epochs2use  
    for kfold in '0' # '1' '2' '3' '4' 
    do
        kfold2use=$kfold
        export kfold2use
       
        # # running commands
        python vae_train.py 
        # python vae_test.py
        # python vae_test_historic.py
        # python vae_plot.py
    done
done

# finished commands
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed