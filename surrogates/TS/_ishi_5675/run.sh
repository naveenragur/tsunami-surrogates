RUNDIR="/mnt/beegfs/nragu/tsunami/japan/Paper/TS/_ishi_5675"
cd $RUNDIR
nepochs2use='5000'
kfold2use='0'
zdim2use='25'
vae2use='100'

# /mnt/beegfs/nragu/tsunami/env/bin/python process_gaugedata.py

for epochs2use in '3000' 
do 
   echo "Run $epochs2use as epoch"
    epochs2use=$epochs2use
    export nepochs2use
    export epochs2use
    export zdim2use
    export vae2use
    
    for kfold in '0' '1' '2' '3' '4' 
    do
        kfold2use=$kfold 
        export kfold2use
        /mnt/beegfs/nragu/tsunami/env/bin/python vae_train.py
        # /mnt/beegfs/nragu/tsunami/env/bin/python vae_test.py
        # /mnt/beegfs/nragu/tsunami/env/bin/python vae_test_historic.py
        # /mnt/beegfs/nragu/tsunami/env/bin/python vae_plot.py
    done
done