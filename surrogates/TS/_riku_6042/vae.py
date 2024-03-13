import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import torch.nn.functional as F
from torchsummary import summary
# from torch.autograd import Variable

np.random.seed(0)
random.seed(0)
torch.random.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1D autoencoder
class Conv1DVAE(nn.Module):
    def __init__(self,ngauges,ninput,zdims):
        super(Conv1DVAE, self).__init__()

        self.ninput = ninput
        self.ngauges = ngauges
        self.zdims = zdims
       
        # encoder
        self.conv1 = nn.Conv1d(self.ninput, 64, 3, padding=1)  #batch size x inputs x npts # output channel # kernel size #padding
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1)
        # self.conv5 = nn.Conv1d(128, 256, 3, padding=1)
        # self.conv6 = nn.Conv1d(256, 256, 3, padding=1)

        # other layers
        self.pool = nn.MaxPool1d(2,2) #pooling size
        self.ht = nn.LeakyReLU(negative_slope=0.5)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p=0.00)
        
        # fully connected layer
        self.fcnfor = nn.Linear(1024*8, self.zdims*2)
        self.fcnback = nn.Linear(self.zdims, 1024*8)

        # decoder
        # self.t_conv6 = nn.ConvTranspose1d(128, 128, 2, stride=2)
        # self.t_conv5 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose1d(128, 128, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(64, 64, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose1d(64, 1, 2, stride=2)

    # def encoder(self, x: Variable) -> (Variable, Variable):
    def encoder(self, x):   
        x = self.ht(self.conv1(x))
        x = self.pool(x)
        # x = self.batchnorm(x)
        x = self.ht(self.conv2(x))
        x = self.pool(x)
        x = self.ht(self.conv3(x))
        x = self.pool(x)
        x = self.ht(self.conv4(x))
        x = self.pool(x)
        # x = self.dropout(x)
        # x = self.ht(self.conv5(x))
        # x = self.pool(x)
        # x = self.ht(self.conv6(x))
        # x = self.pool(x)
        x = x.view(x.shape[0],-1) #x.shape should be batch size
        x = self.fcnfor(x).view(-1, 2, self.zdims)
        mu = x[:, 0, :] # the first feature values as mean
        sig = x[:, 1, :]
        #get dimensions here, output mu and sigma
        return mu, sig

    def decoder(self, x):
        x = self.fcnback(x)
        x = x.view([x.shape[0], 128, 64]) #x.shape[0] is batch size
        # x = self.dropout(x)
        # x = self.ht(self.t_conv6(x))
        # x = self.ht(self.t_conv5(x))
        x = self.ht(self.t_conv4(x))
        x = self.ht(self.t_conv3(x))
        x = self.ht(self.t_conv2(x))
        # x = self.batchnorm(x)
        x = self.ht(self.t_conv1(x))
        # print(x.shape)
        return x

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


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar

class VarAutoEncoder():

    def __init__(self, gauges=[5832],
                 data_path='_data',
                 data_name='riku',
                 model_name='model0'):

        self.gauges = gauges                  #[5832, 5845, 5901, 6042]
        self.ngauges = len(self.gauges)
        
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
                  data_fname=None,
                  kfold=0):
        '''
        load interpolated gauge data 

        set data_fname to designate stored data in .npy format
        
        '''

        device = self.device

        if data_fname == None:
            fname = self.model_name + '.npy'
            data_fname = os.path.join(self.data_path, fname)

        data_all = np.load(data_fname)
        self.nruns = data_all.shape[0]
        model_name = self.model_name
        data_name = self.data_name
        
        # load shuffled indices
        fname = os.path.join(self.data_path,
                            '{:s}_train_index_k{:d}.txt'.format(data_name,kfold))
        train_index = np.loadtxt(fname).astype(int)
        
        fname = os.path.join(self.data_path,
                            '{:s}_test_index_k{:d}.txt'.format(data_name,kfold))
        test_index = np.loadtxt(fname).astype(int)

        data_train = data_all[train_index, : , :]
        data_test  = data_all[test_index, : , :]

        # create a list of batches for training, test sets
        data_train_batch_list = []
        data_test_batch_list = []

        self.batch_size = batch_size
        for i in np.arange(0, data_train.shape[0], batch_size):
            data0 = data_train[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_train_batch_list.append(data0)

        for i in np.arange(0, data_test.shape[0], batch_size):
            data0 = data_test[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_test_batch_list.append(data0)

        self.nbatches_train = len(data_train_batch_list)
        self.nbatches_test  = len(data_test_batch_list)

        self.data_train_batch_list = data_train_batch_list
        self.data_test_batch_list = data_test_batch_list
        
        self.data_fname = data_fname

    def train_vae(self,vaeruns=100,
                        zdims =100,
                        torch_loss_func=nn.MSELoss,
                        torch_optimizer=optim.Adam,
                        nepochs=1000,
                        input_gauges=None,
                        kfold=0, # 60min and 30mins
                        lr=0.0005):
        '''
        Set up and train autoencoder

        vaeruns :
            number of vae runs for the model

        rand_seed :
            seed for randomization 
            (for shuffling data and initializing weights)

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

        data_train_batch_list = self.data_train_batch_list
        data_test_batch_list = self.data_test_batch_list
        output_path = self.output_path
        ngauges = self.ngauges

        # select input gauge
        if input_gauges == None:
            # for default use only the first gauge
            input_gauges = self.gauges[:1]
        
        input_gauges = np.array(input_gauges)
        ninput = len(input_gauges)

        self.input_gauges = input_gauges
        input_gauges_bool = np.array(\
                [np.any(gauge == input_gauges) for gauge in self.gauges])
        self.input_gauges_bool = input_gauges_bool
        ig = np.arange(ngauges)[input_gauges_bool]

        model_name = self.model_name

        self.kfold = kfold
        self.vaeruns = vaeruns
        self.nepochs = nepochs
        self.device = device

        nbatches_train = len(data_train_batch_list)
        nbatches_test = len(data_test_batch_list)

        self.save_model_info()              # save model info
        save_interval = 500 #int(nepochs/30)    # save model every _ epochs

        # define new model
        model = Conv1DVAE(ngauges, ninput, zdims)
        model.to(device)

        # train model
        loss_func = torch_loss_func
        optimizer = torch_optimizer(model.parameters(), lr=lr)

        # epochs
        train_loss_array = np.zeros(nepochs+1)
        test_loss_array = np.zeros(nepochs+1)

        for epoch in range(1, nepochs+1):
            # monitor training loss
            train_loss = 0.0
            test_loss = 0.0

            #Training in batch as k fold cross validation
            for k in range(nbatches_train):
                #train
                data0 = data_train_batch_list[k] #a batch of data
                optimizer.zero_grad() # set to zero for a new batch

                # input is first gauges
                data1 = data0[:,ig,:].detach().clone()

                outputs = model(data1) # prediction of VAE which returns reconstruction, zmean and zvariance or sd 
                loss = loss_func(outputs[0][:,0,:], data0[:,-1,:], outputs[1], outputs[2]) #batch level loss #output gaguge is last gauge
                loss.backward() #computes dloss/dx for every parameter x ie backpropagation
                optimizer.step() #updates the value of x using the gradient x.grad using selected optimizer and learning rate, and loss derivative of this batch
                train_loss += loss.item() #cumalitive epoch loss
                
                #test
                if k < nbatches_test:
                    datatest0 = data_test_batch_list[k]
                    datatest1 = datatest0[:,ig,:].detach().clone()
                    outputstest = model(datatest1)
                    tesloss = F.mse_loss(outputstest[0][:,0,:], datatest0[:,-1,:])
                    test_loss += tesloss.item()
                    
            # avg loss per epoch
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


    def save_model_info(self):

        import pickle

        info_dict = [self.batch_size,
                     self.nbatches_train,
                     self.input_gauges,
                     self.input_gauges_bool,
                     self.nepochs,
                     self.vaeruns,
                     self.gauges,
                     self.ngauges,
                     self.data_path,
                     self.output_path,
                     self.shuffled,
                     self.shuffle_seed,
                     self.init_weight_seed,
                     self.shuffled_batchno,
                     self.model_name,
                     self.data_fname,
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
         self.input_gauges,
         self.input_gauges_bool,
         self.nepochs,
         self.vaeruns,
         self.gauges,
         self.ngauges,
         self.data_path,
         self.output_path,
         self.shuffled,
         self.shuffle_seed,
         self.init_weight_seed,
         self.shuffled_batchno,
         self.model_name,
         self.data_fname,
         self.device] = info_dict

        if device != None:
            self.device = device

        # load data
        self.load_data(batch_size=self.batch_size,
                       data_fname=self.data_fname,
                       kfold=kfold)


    def predict_dataset(self,epoch, kfold, device='cpu'):
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

        data_train_batch_list = self.data_train_batch_list
        data_test_batch_list = self.data_test_batch_list

        nbatches = len(data_train_batch_list) \
                 + len(data_test_batch_list)
        
        ndata_train = sum([data0.shape[0] for 
                           data0 in data_train_batch_list])
        ndata_test  = sum([data0.shape[0] for 
                           data0 in data_test_batch_list])

        data_batch_list = data_train_batch_list \
                        + data_test_batch_list 
            
        ndata = ndata_train + ndata_test

        vaeruns = self.vaeruns

        gauges = np.array(self.gauges)      
        ngauges = self.ngauges
        input_gauges = self.input_gauges
        input_gauges_bool = self.input_gauges_bool

        npts = data_batch_list[0].shape[-1] # TODO: store this somewhere?

        t_unif = np.linspace(0.0, 4.0, npts)

        device = self.device

        model_name = self.model_name
        if epoch == 0:
            epoch = self.nepochs     # use final epoch
        
        # predict dataset

        pred_all = np.zeros((vaeruns, ndata, 1, npts)) #only one output gauge

        for n in range(vaeruns):
            model = self.eval_model(kfold, epoch, device=device)
            k1 = 0
            for k in range(nbatches):  
                msg = '\rTest set, kfold={:d}, model={:6d}, batch={:6d}'\
                        .format(kfold,n,k)
                sys.stdout.write(msg)
                sys.stdout.flush()

                data0 = data_batch_list[k].to(device)

                # setup input data
                datak = data0[:,input_gauges_bool,:].detach().clone()

                kbatch_size = datak.shape[0]

                # evaluate model
                model_out = model(datak)
                if device == 'cpu':
                    model_out = model_out[0].detach().numpy()
                else:
                    model_out = model_out[0].cpu().detach().numpy()
                
                # collect pred & obs
                pred_all[n, k1:(k1 + kbatch_size), :, :] = model_out

                k1+=kbatch_size
                

        fname = '{:s}_{:s}_k{:d}_{:04d}.npy'\
                    .format(model_name, 'test', kfold, epoch)
        save_fname = os.path.join('_output', fname)
        np.save(save_fname, pred_all)



    def eval_model(self, kfold, epoch, device='cpu'):
        r"""
        Returns autoencoder model in evaluation mode

        Parameters
        ----------

        kfold : int
            k fold used

        n : int
            model number in the ensemble
        
        epoch : int
            use model saved after specified number of epochs

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
        self.init_weight_seed += 1
        model.eval()
        return model



    def predict_historic(self, model_input, epoch,kfold, device='cpu'): #for historic events
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
        array of the shape (nensemble, ?, 1 ie the outputgauge, npts)

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
        batch_size = self.batch_size
        vaeruns = self.vaeruns

        gauges = np.array(self.gauges)      
        ngauges = self.ngauges
        input_gauges = self.input_gauges
        input_gauges_bool = self.input_gauges_bool

        npts = model_input.shape[-1]

        model_input = torch.tensor(model_input, dtype=torch.float32)
        t_unif = np.linspace(0.0, 4.0, npts)    # TODO: store this elsewhere?

        device = self.device

        model_name = self.model_name
        if epoch == 0:
            epoch = self.nepochs     # use final epoch
        
        # predict dataset
        pred = np.zeros((vaeruns, 1, npts))
        
        for n in range(vaeruns):

            model = self.eval_model(kfold, epoch, device=device)

            data0 = model_input.to(device)

            # setup input data
            datak = data0[:, input_gauges_bool, :].detach().clone()

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

