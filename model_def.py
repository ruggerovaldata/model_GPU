import torch
from torch import nn

import simulator as s


def RelativeError(true, pred):
  return torch.abs(true-pred)/true


class VariationalAutoEncoder(nn.Module):
  def __init__(self,input_dim,x_size,y_size, nu, nu0,t, h_dim = 500, z_dim_params = 2, z_dim_noise = 500):

    super(VariationalAutoEncoder,self).__init__()
    
    #encoder
    self.img_2hid = nn.Linear(input_dim, h_dim)
    
    self.hid_2mu_params = nn.Linear(h_dim, z_dim_params)
    self.hid_2sigma_params = nn.Linear(h_dim, z_dim_params)

    self.hid_2mu_noise = nn.Linear(h_dim, z_dim_noise)
    self.hid_2sigma_noise = nn.Linear(h_dim, z_dim_noise)

    #decoder: 
    self.zparams_2img = nn.Linear(z_dim_params, input_dim)

    self.hidnoise_2hid = nn.Linear(z_dim_noise,h_dim)
    self.hid_2img = nn.Linear(h_dim, input_dim)

    self.relu = nn.ReLU()

    #Variables for decoder and forward pass 
    self.input_dim = input_dim
    self.nu = nu 
    self.nu0 = nu0
    self.t = t
    self.x_size = x_size
    self.y_size = y_size


  def encode(self,x):
    h = self.relu(self.img_2hid(x))
    mu_params, sigma_params = self.hid_2mu_params(h), self.hid_2sigma_params(h)
    mu_noise, sigma_noise = self.hid_2mu_noise(h), self.hid_2sigma_noise(h)
    return mu_params,sigma_params, mu_noise, sigma_noise
  
  def decodeparams(self,z):
    dm_pred= z[:,0]
    swidth_pred = z[:,1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dm_pred = 10*dm_pred.view(dm_pred.shape[0])
    #swidth_pred = swidth_pred.view(swidth_pred.shape[0])
    swidth_pred = 1000*torch.abs(swidth_pred)
    #print(swidth_pred)
    width = 0.000001*torch.ones(len(dm_pred))
    width = width.to(device)
    out = s.decoder(dm_pred, width,self.nu,self.nu0,self.t,swidth_pred,self.x_size,self.y_size,plot_flag=False)
    return out, [dm_pred,swidth_pred]
  
  def decodenoise(self,z):
    h = self.relu(self.hidnoise_2hid(z))
    img = self.hid_2img(h)
    return img.view(-1,self.y_size,self.x_size)

  def forward(self, x):
    mu_params, logvar_params, mu_noise,logvar_noise = self.encode(x.view(-1,self.input_dim))
    std_params = 0.5*torch.exp(logvar_params-32)
    std_noise = 0.5*torch.exp(logvar_noise-32)
    epsilon_params = std_params.data.new(std_params.size()).normal_()
    epsilon_noise = std_noise.data.new(std_noise.size()).normal_()
    z_reparametrized_params = mu_params + std_params * epsilon_params
    z_reparametrized_noise = mu_noise + std_noise * epsilon_noise
    x_reconstructed_params, params = self.decodeparams(z_reparametrized_params)
    x_reconstructed_noise = self.decodenoise(z_reparametrized_noise)
    return x_reconstructed_params, x_reconstructed_noise, params

class VariationalAutoEncoder_noswidth(nn.Module):
  def __init__(self,input_dim,x_size,y_size, nu, nu0,t, h_dim = 150, h1_dim = 100, z_dim_params = 2, z_dim_noise = 1000):

    super(VariationalAutoEncoder_noswidth,self).__init__()
    
    #encoder
    self.img_2hid = nn.Linear(input_dim, h_dim)

    #self.hid_2hid = nn.Linear(h1_dim,h_dim)
    
    self.hid_2mu_params = nn.Linear(h_dim, z_dim_params)
    self.hid_2sigma_params = nn.Linear(h_dim, z_dim_params)

    self.hid_2mu_noise = nn.Linear(h_dim, z_dim_noise)
    self.hid_2sigma_noise = nn.Linear(h_dim, z_dim_noise)

    #decoder: 
    self.zparams_2img = nn.Linear(z_dim_params, input_dim)

    self.hidnoise_2hid = nn.Linear(z_dim_noise,h_dim)
    self.hid_2img = nn.Linear(h_dim, input_dim)

    self.relu = nn.ReLU()

    #Variables for decoder and forward pass 
    self.input_dim = input_dim
    self.nu = nu 
    self.nu0 = nu0
    self.t = t
    self.x_size = x_size
    self.y_size = y_size


  def encode(self,x):
    h = self.relu(self.img_2hid(x))
    #h = self.relu(self.hid_2hid(h1))
    mu_params, sigma_params = self.hid_2mu_params(h), self.hid_2sigma_params(h)
    mu_noise, sigma_noise = self.hid_2mu_noise(h), self.hid_2sigma_noise(h)
    return mu_params,sigma_params, mu_noise, sigma_noise
  
  def decodeparams(self,z):
    dm_pred= z[:,0]
    swidth_pred = z[:,1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dm_pred = 10*dm_pred.view(dm_pred.shape[0])
    #swidth_pred = swidth_pred.view(swidth_pred.shape[0])
    swidth_pred = 1000*torch.abs(swidth_pred)
    #print(swidth_pred)
    width = 0.00001*torch.ones(len(dm_pred))
    width = width.to(device)
    out = s.decoder_noswidth(dm_pred, width,self.nu,self.nu0,self.t,swidth_pred,self.x_size,self.y_size,plot_flag=False)
    return out, [dm_pred,swidth_pred]
  
  def decodenoise(self,z):
    h = self.relu(self.hidnoise_2hid(z))
    img = self.hid_2img(h)
    return img.view(-1,self.y_size,self.x_size)

  def forward(self, x):
    mu_params, logvar_params, mu_noise,logvar_noise = self.encode(x.view(-1,self.input_dim))
    std_params = 0.5*torch.exp(logvar_params-32)
    std_noise = 0.5*torch.exp(logvar_noise-32)
    epsilon_params = std_params.data.new(std_params.size()).normal_()
    epsilon_noise = std_noise.data.new(std_noise.size()).normal_()
    z_reparametrized_params = mu_params + std_params * epsilon_params
    z_reparametrized_noise = mu_noise + std_noise * epsilon_noise
    x_reconstructed_params, params = self.decodeparams(z_reparametrized_params)
    x_reconstructed_noise = self.decodenoise(z_reparametrized_noise)
    return x_reconstructed_params, x_reconstructed_noise, params

def loss_function(x_hat,x,y_size,x_size):
  x_hat = x_hat.view(x_hat.shape[0],y_size,x_size)
  loss = nn.MSELoss()
  MSE = loss(x_hat,x)
  return MSE

def train(model, epochs, train_dataloader, testing_dataloader, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(0,epochs+1):
        if epoch >0:
            model.train()
            train_loss=0
            average_dms_error = 0
            average_swidth_error = 0 
            for x,dm_obs,swidth_obs in train_dataloader:
                #Forward pass
                dm_obs = dm_obs.to(device)
                swidth_obs = swidth_obs.to(device)
                x = x.to(device)
                x_hat_params, x_hat_noise, temp = model(x.view(x.shape[0],model.x_size*model.y_size))
                batch_dms_avge = torch.sum(RelativeError(dm_obs,temp[0]))
                batch_swidth_avge = torch.sum(RelativeError(swidth_obs,temp[1]))
                loss1 = loss_function(x_hat_params,x, model.y_size, model.x_size)
                loss_noise =  loss_function(x-x_hat_params,x_hat_noise, model.y_size, model.x_size)
                loss = loss1 + 0.1*loss_noise
                train_loss+=loss.item()
                average_dms_error+=batch_dms_avge
                average_swidth_error+=batch_swidth_avge
                #Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Average DMs error in epoch prediction: {:.4f} '.format(average_dms_error/len(train_dataloader)))
            print('Average SWIDTH error in epoch prediction: {:.4f} '.format(average_swidth_error/len(train_dataloader)))
            print(f'===> Epoch {epoch} Average loss: {train_loss / len(train_dataloader):.4f}')
    
            with torch.no_grad():
                model.eval()
                test_loss = 0
                for x,_, _ in testing_dataloader:
                    x = x.to(device)
                    #Forward 
                    x_hat_params, x_hat_noise, temp = model(x.view(x.shape[0],model.x_size*model.y_size))
                    test_loss1 = loss_function(x_hat_params,x, model.y_size, model.x_size)
                    test_loss+=test_loss1+loss_function(x-x_hat_params,x_hat_noise, model.y_size, model.x_size)
                print(f'===>Test loss: {(test_loss-test_loss1):.4f}')
            
    return model

def save_model(model,batch_size):
  from datetime import datetime

  print('Saving')
  now = datetime.now()
  dt_string = now.strftime("%d%m%H:%M")
  model_save_name = 'FRBAEGPU'+str(batch_size)+dt_string+'.pt'
  torch.save(model.state_dict(),model_save_name)

def save_model_real_noise(model,batch_size):
  from datetime import datetime

  print('Saving')
  now = datetime.now()
  dt_string = now.strftime("%d%m%H:%M")
  model_save_name = 'REALNOISE'+str(batch_size)+dt_string+'.pt'
  torch.save(model.state_dict(),model_save_name)

if __name__ == '__main__':
    print('Model ready')