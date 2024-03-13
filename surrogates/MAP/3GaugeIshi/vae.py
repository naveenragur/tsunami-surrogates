import os, sys
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error

np.random.seed(0)
random.seed(0)
torch.random.manual_seed(0)

torch.backends.cudnn.deterministic = True # for reproducibility
torch.backends.cudnn.benchmark = False  

def Gfit(obs, pred): #a normalized least-squares
    obs = np.array(obs)
    pred = np.array(pred)
    Gvalue = 1 - (2*np.sum(obs*pred)/(np.sum(obs**2)+np.sum(pred**2)))
    return Gvalue


# 1D autoencoder
class Conv1DVAE(nn.Module):
    def __init__(self,ninput,noutput,zdims):
        super(Conv1DVAE, self).__init__()

        self.ninput = ninput
        self.noutput = noutput
        self.zdims = zdims
       
        # encoder
        #batch size x inputs x npts # output channel # kernel size #padding
        self.conv1 = nn.Conv1d(self.ninput, 64, 3, padding=1)  #input size - 1024 
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)  
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1)     
        
        self.pool = nn.MaxPool1d(4, 4) #pooling size
        self.ht = nn.LeakyReLU(negative_slope=0.5)
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.1)
        self.fcnfor = nn.Linear(512, self.zdims*2)
        self.fcnback = nn.Linear(self.zdims, self.noutput)
    # def encoder(self, x: Variable) -> (Variable, Variable):
    def encoder(self, x):   
        x = self.ht(self.conv1(x))
        x = self.pool(x)
        x = self.batchnorm(x)
        x = self.ht(self.conv2(x))
        x = self.pool(x)
        x = self.ht(self.conv3(x))
        x = self.pool(x)
        x = self.ht(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.shape[0],-1) #x.shape should be batch size
        # print(x.shape)
        x = self.fcnfor(x).view(-1, 2, self.zdims)
        mu = x[:, 0, :] # the first feature values as mean
        sig = x[:, 1, :]
        #get dimensions here, output mu and sigma
        return mu, sig

    def reparameterize(self, mu, logvar): 
        """The reparameterization is key to minimizing posteriors/priors

        For each training sample (we get some number batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [some dim, ZDIMS] mean matrix
        logvar : [som dim, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.
        """
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decoder(self, x):
        x = self.fcnback(x)
        x = x.view(x.shape[0], 1,-1)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar

class VarAutoEncoder():

    def __init__(self, gauges=[5832],
                 noutput=6648,
                 data_path='_data',
                 data_name='riku',
                 model_name='model0'):

        self.gauges = gauges                 
        self.ninput = len(self.gauges)
        self.noutput = noutput
        self.data_path = data_path

        if not os.path.exists('_output'):
            os.mkdir('_output')
        self.output_path = '_output'
        
        self.data_fname = None
        self.data_name = data_name

        self.shuffled = None
        self.shuffle_seed = 0
        self.init_weight_seed = 0

        self.model_name = model_name
        self.device='cpu'   # set to gpu if needed

        self.input_gauges_bool = None

        self.shuffled_batchno = False
        self.data_train_batch_list = None
        self.test_train_batch_list = None 

        self.use_Agg = True

        # set dpi for plt.savefig
        self._dpi = 300

    def load_data(self,
                  batch_size=20,
                  data_fname_in=None,
                  data_fname_out=None,
                  kfold=0):
        '''
        load interpolated gauge data 

        set data_fname to designate stored data in .npy format
        
        '''

        device = self.device
        print('device = ', device)
        if data_fname_in == None:
            fname_in = self.data_name + '_gaugeinputs.npy'
            fname_out = self.data_name + '_floodinputs.npy'
            data_fname_in = os.path.join(self.data_path, fname_in)
            data_fname_out = os.path.join(self.data_path, fname_out)

        data_in = np.load(data_fname_in)
        data_out = np.load(data_fname_out)
        
        self.nruns = data_in.shape[0]
        model_name = self.model_name
        data_name = self.data_name
        
        # load shuffled indices
        fname = os.path.join(self.data_path,
                            '{:s}_train_index_k{:d}.txt'.format(data_name,kfold))
        train_index = np.loadtxt(fname).astype(int)
        
        fname = os.path.join(self.data_path,
                            '{:s}_test_index_k{:d}.txt'.format(data_name,kfold))
        test_index = np.loadtxt(fname).astype(int)

        data_train_in = data_in[train_index, : , :]
        data_test_in  = data_in[test_index, : , :]
        data_train_out = data_out[train_index, : , :]
        data_test_out  = data_out[test_index, : , :]


        # create a list of batches for training, test sets
        data_train_batch_list_in = []
        data_train_batch_list_out = []
        data_test_batch_list_in = []
        data_test_batch_list_out = []

        self.batch_size = batch_size
        for i in np.arange(0, data_train_in.shape[0], batch_size):
            data0 = data_train_in[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_train_batch_list_in.append(data0)
        
        for i in np.arange(0, data_train_out.shape[0], batch_size):
            data0 = data_train_out[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_train_batch_list_out.append(data0)

        for i in np.arange(0, data_test_in.shape[0], batch_size):
            data0 = data_test_in[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_test_batch_list_in.append(data0)
        
        for i in np.arange(0, data_test_out.shape[0], batch_size):
            data0 = data_test_out[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_test_batch_list_out.append(data0)

        self.nbatches_train = len(data_train_batch_list_in)
        self.nbatches_test  = len(data_test_batch_list_in)

        self.data_train_batch_list_in = data_train_batch_list_in
        self.data_test_batch_list_in = data_test_batch_list_in
        self.data_train_batch_list_out = data_train_batch_list_out
        self.data_test_batch_list_out = data_test_batch_list_out

        self.data_fname_in = data_fname_in
        self.data_fname_out = data_fname_out
        
    def train_vae(self,vaeruns=100,
                        zdims =100,
                        torch_loss_func=nn.MSELoss,
                        torch_optimizer=optim.Adam,
                        nepochs=1000,
                        kfold=0,
                        lr=0.0005,
                        weight_decay=0):
        '''
        Set up and train autoencoder

        vaeruns :
            number of vae runs for the model

        rand_seed :
            seed for randomization 
            (for shuffling data and initializing weights)

        zdim :
            no of latent dimensions
        
        kfold :
            fold to load train and test data
        input_gauges :
            gauges to use as inputs, sublist of gauges

        torch_loss_func :
            pytorch loss function, default is torch.nn.MSELoss

        torch_optimizer :
            pytorch loss function, default is torch.nn.MSELoss

        '''

        # set random seed
        init_weight_seed = self.init_weight_seed
        np.random.seed(init_weight_seed)
        random.seed(init_weight_seed)
        torch.random.manual_seed(init_weight_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = self.device

        data_train_batch_list_in = self.data_train_batch_list_in
        data_test_batch_list_in = self.data_test_batch_list_in
        data_train_batch_list_out = self.data_train_batch_list_out
        data_test_batch_list_out = self.data_test_batch_list_out
        output_path = self.output_path
        ninput = self.ninput
        noutput = self.noutput

        model_name = self.model_name
        self.kfold = kfold
        self.vaeruns = vaeruns
        self.nepochs = nepochs
        self.device = device

        nbatches_train = len(data_train_batch_list_in)
        nbatches_test = len(data_test_batch_list_in)

        self.save_model_info()              # save model info
        save_interval = 1000 #int(nepochs/30)    # save model every _ epochs

        # define new model
        model = Conv1DVAE(ninput, noutput, zdims)
        model.to(device)

        # train model
        loss_func = torch_loss_func
        optimizer = torch_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

        # epochs
        train_loss_array = np.zeros(nepochs+1)
        test_loss_array = np.zeros(nepochs+1)

        for epoch in range(1, nepochs+1):
            # monitor training loss
            train_loss = 0.0
            test_loss = 0.0

            #Training in batch as k fold cross validation
            for k in range(nbatches_train):
                #training
                data_in = data_train_batch_list_in[k]
                data_out = data_train_batch_list_out[k]
                optimizer.zero_grad()
                
                outputs = model(data_in)
                loss = loss_func(outputs[0], data_out, outputs[1], outputs[2])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                    
                #test
                if k < nbatches_test: 
                    data_in = data_test_batch_list_in[k]
                    data_out = data_test_batch_list_out[k]

                    outputstest = model(data_in)
                    tesloss = F.mse_loss(outputstest[0], data_out)
                    test_loss += tesloss.item()
                    
            # avg training\test loss per epoch
            avg_train_loss = train_loss/nbatches_train
            train_loss_array[epoch] = avg_train_loss 

            avg_test_loss = test_loss/nbatches_test
            test_loss_array[epoch] = avg_test_loss 

            msg = '\repoch = {:4d}, trainloss = {:1.8f}, testloss = {:1.8f}'.format(epoch,avg_train_loss,avg_test_loss)
            sys.stdout.write(msg)
            sys.stdout.flush()

            fname = '{:s}_train_loss_k{:d}.npy'\
                    .format(model_name, kfold)
            save_fname = os.path.join(output_path, fname)
            np.save(save_fname, train_loss_array)

            fname = '{:s}_test_loss_k{:d}.npy'\
                    .format(model_name, kfold)
            save_fname = os.path.join(output_path, fname)
            np.save(save_fname, test_loss_array)

            if ((epoch) % save_interval) == 0:# and avg_train_loss < 0.1:
                # save intermediate model
                fname ='{:s}_k{:d}_{:04d}.pkl'\
                        .format(model_name, kfold, epoch)
                save_fname = os.path.join(output_path, fname)
                torch.save(model, save_fname)

            if epoch == 1:
                min_loss = avg_test_loss
                min_loss_epoch = epoch
           
            if avg_test_loss < min_loss:
                min_loss = avg_test_loss
                min_loss_epoch = epoch
                fname ='{:s}_k{:d}_{:04d}.pkl'\
                        .format(model_name, kfold, 0)
                save_fname = os.path.join(output_path, fname)
                torch.save(model, save_fname)

            if epoch == nepochs:
                print('\nmin loss = ', min_loss, 'at epoch = ', min_loss_epoch)
            
            #check for early stopping if min_loss_epoch is 1000 epochs ago the current epoch
            if epoch - min_loss_epoch > 10000:
                print('\nearly stopping at epoch = ', epoch, 'min loss epoch = ', min_loss_epoch, 'min loss = ', min_loss)
                break

    def save_model_info(self):

        import pickle

        info_dict = [self.batch_size,
                     self.nbatches_train,
                     self.nepochs,
                     self.vaeruns,
                     self.gauges,
                     self.ninput,
                     self.noutput,
                     self.data_path,
                     self.output_path,
                     self.shuffled,
                     self.shuffle_seed,
                     self.init_weight_seed,
                     self.shuffled_batchno,
                     self.model_name,
                     self.data_fname_in,
                     self.data_fname_out,
                     self.device]
        
        fname = '{:s}_info.pkl'.format(self.model_name)
        save_fname = os.path.join(self.output_path, fname)
        pickle.dump(info_dict, open(save_fname,'wb'))
        

    def load_model(self,model_name,kfold,device=None):

        import pickle

        # load model info
        fname = '{:s}_info.pkl'.format(model_name)
        load_fname = os.path.join(self.output_path, fname)
        info_dict = pickle.load(open(load_fname,'rb'))

        [self.batch_size,
        self.nbatches_train,
        self.nepochs,
        self.vaeruns,
        self.gauges,
        self.ninput,
        self.noutput,
        self.data_path,
        self.output_path,
        self.shuffled,
        self.shuffle_seed,
        self.init_weight_seed,
        self.shuffled_batchno,
        self.model_name,
        self.data_fname_in,
        self.data_fname_out,
        self.device] = info_dict

        if device != None:
            self.device = device

        # load data
        self.load_data(batch_size=self.batch_size,kfold=kfold)

    def predict_dataset(self,epoch,kfold,vae2use=None,device='cpu'):
        r"""
        Predict all of the data set, both training and test sets

        Parameters
        ----------

        epoch : int
            use model after training specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Notes
        -----
        the prediction result is stored as binary numpy arrays in 
        the output directory

        """

        # load data and data dimensions
        batch_size = self.batch_size

        data_train_batch_list_in = self.data_train_batch_list_in
        data_test_batch_list_in = self.data_test_batch_list_in

        nbatches = len(data_train_batch_list_in) \
                 + len(data_test_batch_list_in)
        
        ndata_train = sum([data0.shape[0] for 
                           data0 in data_train_batch_list_in])
        ndata_test  = sum([data0.shape[0] for 
                           data0 in data_test_batch_list_in])

        data_batch_list_in = data_train_batch_list_in \
                        + data_test_batch_list_in 
        
        ndata = ndata_train + ndata_test
        if vae2use is None:
            vaeruns = self.vaeruns
        else:
            vaeruns = vae2use
        noutput = self.noutput
        device = self.device

        model_name = self.model_name
        if epoch == None:
            epoch = self.nepochs     # use final epoch
        
        # predict dataset
        pred_all = np.zeros((vaeruns, ndata, 1, noutput))
        # z_all = np.zeros((vaeruns,ndata, 10))
        for n in range(vaeruns):
            model = self.eval_model(kfold, epoch, device=device) #with new initialized seed see eval_model

            k1 = 0

            for k in range(nbatches):  
                msg = '\rTest set, kfold={:d}, model={:6d}, batch={:6d}'\
                        .format(kfold,n,k)
                sys.stdout.write(msg)
                sys.stdout.flush()

                datak = data_batch_list_in[k].to(device)
                kbatch_size = datak.shape[0]

                # evaluate model
                model_out = model(datak)
                # mu_out, logvar_out = model.encoder(datak)
                if device == 'cpu':
                    model_out = model_out[0].detach().numpy()
                    # mu_out = mu_out.detach().numpy()
                else:
                    model_out = model_out[0].cpu().detach().numpy()
                    # mu_out = mu_out.cpu().detach().numpy()
                
                # collect pred & obs
                pred_all[n, k1:(k1 + kbatch_size), :] = model_out
                # z_all[n, k1:(k1 + kbatch_size),:] = mu_out

                k1+=kbatch_size
                    
        fname = '{:s}_{:s}_k{:d}_{:04d}.npy'\
        .format(model_name, 'test', kfold, epoch)
        save_fname = os.path.join('_output', fname)
        print('\nsaving to ', save_fname)
        np.save(save_fname, pred_all)

        # fname = '{:s}_{:s}_k{:d}_{:04d}.npy'\
        #             .format(model_name, 'test_z', kfold, epoch)
        # save_fname = os.path.join('_output', fname)
        # np.save(save_fname, z_all)

    def eval_model(self,kfold, epoch, device='cpu'):
        r"""
        Returns autoencoder model in evaluation mode

        Parameters
        ----------

        kfold : int
            k fold used
        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Returns
        -------
        
        model : Conv1d module
            Conv1d module in evaluation mode

        """
        model_name = self.model_name

        # load stored autoencoder
        fname = '{:s}_k{:d}_{:04d}.pkl'\
                .format(model_name, kfold, epoch)

        load_fname = os.path.join('_output', fname)
        model = torch.load(load_fname,
                           map_location=torch.device(device))

        # set random seed
        init_weight_seed = self.init_weight_seed
        np.random.seed(init_weight_seed)
        random.seed(init_weight_seed)
        torch.random.manual_seed(init_weight_seed)
        
        # increment seed
        self.init_weight_seed += 10

        model.eval()

        return model

    def predict_historic(self, model_input, epoch,kfold,vae2use=None,device='cpu'):
        r"""
        Predict all of the data set, both training and test sets

        Parameters
        ----------

        model_input : tensor
            model input of size (?, ninput_gauges, npts)

        epoch : int
            use model after training specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Returns
        -------
        
        pred : list of arrays
            prediction results for each prediction kfold, each item is a numpy
        array of the shape (nensemble, ?, 1 ie the location, noutpts)


        Notes
        -----

        items in output pred is also saved to output directory in binary 
        numpy array format

        """

        # set random seed
        init_weight_seed = self.init_weight_seed
        np.random.seed(init_weight_seed)
        random.seed(init_weight_seed)
        torch.random.manual_seed(init_weight_seed)

        # load data and data dimensions
        if vae2use is None:
            vaeruns = self.vaeruns
        else:
            vaeruns = vae2use
     
        noutput = self.noutput
        ndata = model_input.shape[0]

        model_input = torch.tensor(model_input, dtype=torch.float32)
        device = self.device

        model_name = self.model_name
        if epoch == None:
            epoch = self.nepochs     # use final epoch
        

        # predict dataset
        pred = np.zeros((vaeruns, ndata, noutput))
        
        for n in range(vaeruns):

            model = self.eval_model(kfold, epoch, device=device) #with new initialized seed see eval_model

            datak = model_input.to(device)

            # evaluate model
            model_out = model(datak)
            if device == 'cpu':
                model_out = model_out[0].detach().numpy()
            else:
                model_out = model_out[0].cpu().detach().numpy()

            # collect predictions
            pred[n, ...] = model_out
        #print model architecture
        # summary(model,(1,1024))
        # save output to _output
        # fname = '{:s}_{:s}_k{:d}_{:02d}_{:04d}.npy'\
        #             .format(model_name, 'historic', kfold, n, epoch)
        # save_fname = os.path.join('_output', fname)
        # np.save(save_fname, pred)

        return pred

def load_obspred(model_name,kfold,epoch):
    r"""
    Plot time-series prediction

    Parameters
    ----------

    model_name : string
        name of the model. will load results with the filename

        '{:s}_test_{:02d}_{:04d}.npy'.format(model_name, obs_win, epoch)

        under the _output directory

    epochs_list : list of ints
        list of ints choosing the training epoch (for each obs window)

    Returns
    -------

    pred_train : array-like
        prediction results for the training set, of the shape
        (len(obs_win_list), nensemble, ndata_train, ngauges, npts)

    obs_train : array-like
        observations in the training set, of the shape
        (ndata_train, ngauges, npts)

    pred_test : array-like
        prediction results for the training set, of the shape
        (len(obs_win_list), nensemble, ndata_test, ngauges, npts)

    obs_test : array-like
        observations in the test set, of the shape
        (ndata_test, ngauges, npts)

    train_runno : array-like
        Geoclaw run numbers of the data in the training set

    test_runno : array-like
        Geoclaw run numbers of the data in the test set

    """

    # load all observed data
    data_dir = '_data'
    fname = os.path.join(data_dir, 'riku_floodinputs.npy')

    obs = np.load(fname)

    # load shuffled indices
    fname = os.path.join(data_dir, 'riku_train_index_k{:d}.txt'.format(kfold))
    train_index = np.loadtxt(fname).astype(np.int64)

    fname = os.path.join(data_dir, 'riku_test_index_k{:d}.txt'.format(kfold))
    test_index = np.loadtxt(fname).astype(np.int64)

    fname = os.path.join(data_dir, 'riku_train_runno_k{:d}.txt'.format(kfold))
    train_runno = np.loadtxt(fname).astype(np.int64)

    fname = os.path.join(data_dir, 'riku_test_runno_k{:d}.txt'.format(kfold))
    test_runno = np.loadtxt(fname).astype(np.int64)

    obs_train = obs[train_index, :, :]
    obs_test = obs[test_index, :, :]


    # prediction
    pred_train = []
    pred_test = []
    # Prediction vs. Observation plot

    # load prediction
    
    fname = '{:s}_test_k{:d}_{:04d}.npy'.format(model_name,kfold,epoch)
    load_fname = os.path.join('_output', fname)
    pred_all = np.load(load_fname)

    pred_train = pred_all[:, :len(train_index), :, :]
    pred_test = pred_all[:, -len(test_index):, :, :]

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)

    return pred_train, obs_train, pred_test, obs_test, train_runno, test_runno


def plot_obsvpred(pred_train, obs_train,
                  pred_test, obs_test,
                  scale=1.0):
    r"""
    Plot prediction vs. observation
    np.ceil
    Parameters
    ----------

    pred_train : array-like
        ensemble prediction time-series for train set with shape 
        (nensemble, ndata_train, ngauges, npts)

    obs_train : array-like
        time-series observation for test data
        (ndata_train, ngauges, npts)

    pred_test : array-like
        ensemble prediction time-series for test set with shape 
        (nensemble, ndata_test, ngauges, npts)

    obs_test : array-like
        time-series observation for test data
        (ndata_test, ngauges, npts)

    Returns
    -------
    fig_list : list
        a list of tuples containing figure, axis, gauge number

    """
    nensemble, ndata_train, ngauges, npts = pred_train.shape
    #(100, 392, 1, 6648) pred_train shape
    #(392, 1, 6648) obs_train shape

    pred_train_etamax = pred_train.max(axis=-1)
    pred_test_etamax = pred_test.max(axis=-1)

    obs_train_etamax = obs_train.max(axis=-1)
    obs_test_etamax = obs_test.max(axis=-1)

    predmean_train = np.mean(pred_train_etamax, axis=0)
    predmean_test = np.mean(pred_test_etamax, axis=0)

    print(pred_train_etamax.shape)
    pred2std_train = 2.0 * np.std(pred_train_etamax, axis=0)
    pred2std_test = 2.0 * np.std(pred_test_etamax, axis=0)

    fig_list = []

    # make a plot for each location
    for i in range(ngauges):

        obs_train_gaugei = obs_train_etamax[:, i]
        obs_test_gaugei = obs_test_etamax[:, i]

        predmean_train_gaugei = predmean_train[:, i]
        predmean_test_gaugei = predmean_test[:, i]

        pred2std_train_gaugei = pred2std_train[:, i]
        pred2std_test_gaugei = pred2std_test[:, i]

        vmax = max(predmean_train_gaugei.max(),
                   predmean_test_gaugei.max(),
                   obs_train_gaugei.max(),
                   obs_test_gaugei.max())

        fig, ax = plt.subplots(figsize=(scale * 4, scale * 4))

        if vmax > 10.0:
            vmax = 20.0
        else:
            vmax = 15.0

        ax.plot([0.0, 1.10 * vmax],
                [0.0, 1.10 * vmax],
                '-k',
                linewidth=0.5)

        # plot obs/pred in train set
        line0, = ax.plot(obs_train_gaugei,
                         predmean_train_gaugei,
                         'k.',
                         alpha=0.3,
                         markersize=2,
                         zorder=0)

        # plot obs/pred in test set
        line1, = ax.plot(obs_test_gaugei,
                         predmean_test_gaugei,
                         linewidth=0,
                         marker='D',
                         color='tab:orange',
                         markersize=1.5,
                         zorder=4)

        # plot errorbars for predictions in test set
        ax.errorbar(obs_test_gaugei,
                    predmean_test_gaugei,
                    yerr=pred2std_test_gaugei,
                    fmt='.',
                    color='tab:orange',
                    elinewidth=0.8,
                    alpha=0.3,
                    capsize=0,
                    markersize=2)

        ax.text(10, 3.5, '$MSE Train$:' + str(round(mean_squared_error(obs_train_gaugei, predmean_train_gaugei), 3)), fontsize=8)
        ax.text(10, 2.5, '$MSE Test$:' + str(round(mean_squared_error(obs_test_gaugei, predmean_test_gaugei), 3)), fontsize=8)
        ax.text(10, 1.5, '$Gfit Train$:' + str(round(Gfit(obs_train_gaugei, predmean_train_gaugei), 3)), fontsize=8)
        ax.text(10, 0.5, '$Gfit Test$:' + str(round(Gfit(obs_test_gaugei, predmean_test_gaugei), 3)), fontsize=8)

        legends = [[line0, line1],
                   ['train'+'('+ str(round(r2_score(obs_train_gaugei, predmean_train_gaugei), 3))+')',
                    'test'+'('+ str(round(r2_score(obs_test_gaugei, predmean_test_gaugei), 3))+')']]

        ax.set_xlabel('observed $\eta_{max}$ ($m$)')
        ax.set_ylabel('predicted $\eta_{max}$ ($m$)')
        ax.grid(True, linestyle=':')
        ax.set_aspect('equal')

        ax.set_xlim([0.0, vmax])
        ax.set_ylim([0.0, vmax])

        if vmax > 10:
            ax.xaxis.set_ticks(np.arange(0.0, vmax + 1, 2))
            ax.yaxis.set_ticks(np.arange(0.0, vmax + 1, 2))
        else:
            ax.xaxis.set_ticks(np.arange(0.0, vmax + 1))
            ax.yaxis.set_ticks(np.arange(0.0, vmax + 1))

        fig_list.append((fig, ax, legends))

    return fig_list

