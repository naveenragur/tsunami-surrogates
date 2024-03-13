RUNDIR="$TSU/ML_Tohoku/Paper/_shio_5440"
cd $RUNDIR
nepochs2use='5000'
kfold2use='0'
zdim2use='100'
vae2use='100'

# /home/nragu/opt/anaconda3/envs/pytorch/bin/python process_gaugedata.py

for epochs2use in '1500' 
do 
   echo "Run $epochs2use as epoch"
    epochs2use=$epochs2use
    export nepochs2use
    export epochs2use
    export zdim2use
    export vae2use
    
    for kfold in '0' #'1' '2' '3' '4'  
    do
        kfold2use=$kfold 
        export kfold2use
        # /home/nragu/opt/anaconda3/envs/pytorch/bin/python vae_train.py
        /home/nragu/opt/anaconda3/envs/pytorch/bin/python vae_test.py
        /home/nragu/opt/anaconda3/envs/pytorch/bin/python vae_test_historic.py
        /home/nragu/opt/anaconda3/envs/pytorch/bin/python vae_plot.py
    done
done