import numpy as np 
import pandas as pd
import os
import sys

from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from readsettings import ReadSettings
from torchsummary import summary

# Load and preprocess the data

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
def preprocess(data, seq_len):
    ori_data = data[::-1]
    scaler = MinMaxScaler().fit(ori_data)
    ori_data = scaler.transform(ori_data)
    
    temp_data = []
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
        
    idx = np.random.permutation(len(temp_data))
    data = [temp_data[i] for i in idx]
    return np.array(data)


class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index])

###### TimeGan classes
# 
#

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = torch.sigmoid(self.fc(out))
        return out

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.sigmoid(self.fc(out))
        return out

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Embedder, self).__init__()
        self.model = GRUNet(input_dim, hidden_dim, hidden_dim, n_layers)
    
    def forward(self, x):
        return self.model(x)

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers):
        super(Recovery, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, output_dim, n_layers)
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(Generator, self).__init__()
        self.model = GRUNet(input_dim, hidden_dim, hidden_dim, n_layers)
    
    def forward(self, x):
        return self.model(x)

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(Supervisor, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, hidden_dim, n_layers)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        super(Discriminator, self).__init__()
        self.model = GRUNet(hidden_dim, hidden_dim, 1, n_layers)
    
    def forward(self, x):
        return self.model(x)

class TimeGAN:
    def __init__(self, input_dim, hidden_dim, n_layers, seq_len, n_seq, gamma, batch_size, lr):
        
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.gamma = gamma
        self.batch_size = batch_size
        
        print('input_dim:',input_dim)
        print('hidden_dim:',hidden_dim)
        print('n_layers:',n_layers)
        print('n_seq:',n_seq)
        

        self.embedder = Embedder(input_dim, hidden_dim, n_layers).to(device)
        self.recovery = Recovery(hidden_dim, input_dim, n_layers).to(device)
        self.generator = Generator(n_seq, hidden_dim, n_layers).to(device)
        self.supervisor = Supervisor(hidden_dim, n_layers).to(device)
        self.discriminator = Discriminator(hidden_dim, n_layers).to(device)

        self.autoencoder = nn.Sequential(self.embedder, self.recovery).to(device)

        self.opt_embedder = optim.Adam(self.embedder.parameters(), lr=lr)
        self.opt_recovery = optim.Adam(self.recovery.parameters(), lr=lr)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_supervisor = optim.Adam(self.supervisor.parameters(), lr=lr)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr)

        self.criterion_mse = nn.MSELoss()
        self.criterion_bce = nn.BCELoss()

    def train_autoencoder(self, x):
        self.opt_embedder.zero_grad()
        self.opt_recovery.zero_grad()

        h = self.embedder(x)
        x_tilde = self.recovery(h)
        embedding_loss_t0 = self.criterion_mse(x, x_tilde)
        e_loss_0 = 10 * torch.sqrt(embedding_loss_t0)

        e_loss_0.backward()
        self.opt_embedder.step()
        self.opt_recovery.step()

        return torch.sqrt(embedding_loss_t0)

    def train_supervisor(self, x):
        self.opt_supervisor.zero_grad()
        h = self.embedder(x)
        h_hat_supervised = self.supervisor(h)
        g_loss_s = self.criterion_mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        g_loss_s.backward()
        self.opt_supervisor.step()

        return g_loss_s

    def train_embedder(self, x):
        self.opt_embedder.zero_grad()
        self.opt_recovery.zero_grad()

        h = self.embedder(x)
        h_hat_supervised = self.supervisor(h)
        generator_loss_supervised = self.criterion_mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_tilde = self.autoencoder(x)
        embedding_loss_t0 = self.criterion_mse(x, x_tilde)
        e_loss = 10 * torch.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        e_loss.backward()
        self.opt_embedder.step()
        self.opt_recovery.step()

        return torch.sqrt(embedding_loss_t0)

    def discriminator_loss(self, x, z):
        y_real = self.discriminator(self.embedder(x))
        discriminator_loss_real = self.criterion_bce(y_real, torch.ones_like(y_real))

        y_fake = self.discriminator(self.supervisor(self.generator(z)))
        discriminator_loss_fake = self.criterion_bce(y_fake, torch.zeros_like(y_fake))

        y_fake_e = self.discriminator(self.generator(z))
        discriminator_loss_fake_e = self.criterion_bce(y_fake_e, torch.zeros_like(y_fake_e))

        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = torch.mean(y_true, dim=0), torch.var(y_true, dim=0)
        y_pred_mean, y_pred_var = torch.mean(y_pred, dim=0), torch.var(y_pred, dim=0)
        g_loss_mean = torch.mean(torch.abs(y_true_mean - y_pred_mean))
        g_loss_var = torch.mean(torch.abs(torch.sqrt(y_true_var + 1e-6) - torch.sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    def train_generator(self, x, z):
        self.opt_generator.zero_grad()
        self.opt_supervisor.zero_grad()

        y_fake = self.discriminator(self.supervisor(self.generator(z)))
        generator_loss_unsupervised = self.criterion_bce(y_fake, torch.ones_like(y_fake))

        y_fake_e = self.discriminator(self.generator(z))
        generator_loss_unsupervised_e = self.criterion_bce(y_fake_e, torch.ones_like(y_fake_e))

        h = self.embedder(x)
        h_hat_supervised = self.supervisor(h)
        generator_loss_supervised = self.criterion_mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        x_hat = self.recovery(self.supervisor(self.generator(z)))
        generator_moment_loss = self.calc_generator_moments_loss(x, x_hat)

        generator_loss = (generator_loss_unsupervised +
                          generator_loss_unsupervised_e +
                          100 * torch.sqrt(generator_loss_supervised) +
                          100 * generator_moment_loss)

        generator_loss.backward()
        self.opt_generator.step()
        self.opt_supervisor.step()

        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    def train_discriminator(self, x, z):
        self.opt_discriminator.zero_grad()
        discriminator_loss = self.discriminator_loss(x, z)
        discriminator_loss.backward()
        self.opt_discriminator.step()
        return discriminator_loss

    def get_batch_data(self, data, n_windows):
        data_loader = DataLoader(StockDataset(data), batch_size=self.batch_size, shuffle=True)
        return iter(data_loader)

    def _generate_noise(self):
            return np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))

    def get_batch_noise(self):
        noise_loader = DataLoader(list(self._generate_noise()), batch_size=self.batch_size)
        #print("noise",len(noise_loader))
        return iter(noise_loader)

    def sample(self, n_samples):
        steps = n_samples #// self.batch_size + 1
        data = []
        print('Steps:',steps)
        for _ in trange(steps, desc='Synthetic data generation'):
            Z_ = next(self.get_batch_noise()).float()
            x_h = self.recovery(self.generator(Z_.to(device))).cpu().detach().numpy()
            data.append(x_h)
        return np.array(np.vstack(data))        
######################## main  
#       
def main():
    args = sys.argv[1:]
    print(args)
    data = ReadSettings(args[0])
    dataset_path = data["paths"]["dataset"]
    path_output = data["paths"]["output"]
    
    seq_len = data["model_parameters"]["seq_len"]
    n_seq = data["model_parameters"]["n_seq"]
    hidden_dim = data["model_parameters"]["hidden_dim"]
    gamma = data["model_parameters"]["gamma"]
    
    noise_dim = data["model_parameters"]["noise_dim"]
    dim = data["model_parameters"]["dim"]
    batch_size = data["model_parameters"]["batch_size"]
    
    log_step = data["model_parameters"]["log_step"]
    learning_rate = data["model_parameters"]["learning_rate"]
    train_steps = data["model_parameters"]["train_steps"]

    sample_size = data["model_parameters"]["sample_size"]
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' 
    batch_size = 128
    learning_rate = 5e-4 
    noise_dim = 32
    dim = 128
    input_dim = 6
    model_parameters = (batch_size, learning_rate, 0.5, 0.5, noise_dim, 24, 2, (0, 1), dim)
    hidden_dim = 24
    seq_len = 24
    n_layers = 3
    n_seq = 6
    gamma = 1
    train_steps = 1#5000
    sample_size = 3660
    
    '''
    df = pd.read_csv('../../../dataset/TimeGan/Google_BIG.csv')
    input_dim = n_seq
    n_layers = 3
    model_parameters = (batch_size, learning_rate, 0.5, 0.5, noise_dim, 24, 2, (0, 1), dim)
    stock_data = preprocess(df.values, seq_len)
    synth = TimeGAN(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers, seq_len=seq_len, n_seq=n_seq, gamma=gamma, batch_size=batch_size, lr=learning_rate)

    for _ in tqdm(range(train_steps), desc='Embedding network training'):
        X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
        #print("-----------------Shape:",X_.shape)
        step_e_loss_t0 = synth.train_autoencoder(X_.to(device))
        #print('step_e_loss_t0 ', step_e_loss_t0)

    for _ in tqdm(range(train_steps), desc='Supervised network training'):
        X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
        #print("-----------------Shape:",X_.shape)
        step_g_loss_s = synth.train_supervisor(X_.to(device))
        #print('step_g_loss_s  ', step_g_loss_s)

    step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0

    for _ in tqdm(range(train_steps), desc='Joint networks training'):
        for _ in range(1):
            X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
            Z_ = next(synth.get_batch_noise()).float()
            #print('XXX  ', X_.shape)
            #print('ZZZZ  ', Z_.shape)
            #print('ZZ type ', type(torch.FloatTensor(Z_)))
            step_g_loss_u, step_g_loss_s, step_g_loss_v = synth.train_generator(X_.to(device),Z_.to(device))
            #print("-----------------Shape:",X_.shape)
            step_e_loss_t0 = synth.train_embedder(X_.to(device))
            #print('step_e_loss_t0 ', step_e_loss_t0)

        X_ = next(synth.get_batch_data(stock_data, n_windows=len(stock_data)))
        Z_ = next(synth.get_batch_noise()).float()
        step_d_loss = synth.discriminator_loss(X_.to(device), Z_.to(device))
        #print('step_d_loss  ', step_d_loss)
        if step_d_loss > 0.15:
            step_d_loss = synth.train_discriminator(X_.to(device), Z_.to(device))
    print(synth.embedder)
    #summary(synth.embedder,(24,6))
    #summary(synth.embedder,input_size=(128,24,6))

    print(synth.recovery)
    print(synth.generator)
    print(synth.supervisor)
    print(synth.discriminator)

    torch.save(synth.embedder,path_output+"embedder.pt")
    torch.save(synth.recovery,path_output+"recovery.pt")
    torch.save(synth.generator,path_output+"generator.pt")
    torch.save(synth.supervisor,path_output+"supervisor.pt")
    torch.save(synth.discriminator,path_output+"discriminator.pt")

    ############################################################

   # synthetic_sample = np.asarray(synth_data)

   # print(real_sample.shape)
   # print(synth_data.shape)

if __name__ == '__main__':
    main()