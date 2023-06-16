import torch
from torch.utils.data import Dataset, DataLoader

import simulator as s

import argparse 
import matplotlib.pyplot as plt

""" parser = argparse.ArgumentParser()

parser.add_argument('x_size', type=float, help='x_size of the images to be generated')
parser.add_argument('y_size', type=float, help='y_size of the images to be generated')
parser.add_argument('nu_i', type=float, help='Lower value of the frequency channels')
parser.add_argument('nu_f', type=float, help='Higher frequency channel')
parser.add_argument('t', type=float, help='Time from the central value of time')
parser.add_argument('size', type=float, help='Size of the dataset')
parser.add_argument('train_per',type=float, help='percentage to divide the dataset in training and testing')
parser.add_argument('DM_min', type=float, help='Minimum of the DM')
parser.add_argument('DM_max', type=float, help='maximum of the DM')

args = parser.parse_args() """

#n_tot = args.size 


class SimulatedDataset(Dataset):
    def __init__(self, images, dm, swidth):
        self.sample=[]
        for i, val in enumerate(images):
          self.sample.append((val,dm[i],swidth[i]))

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]

def rand_interv(a,b,n):
  u = torch.rand(n)
  return (a-b)*u+b 

def CreateDataset(DM_min,DM_max, nu_i,nu_f, N, t, x_size, y_size, batch_size, noise = 7, test_flag=False):

    nu = torch.linspace(nu_i, nu_f, y_size)
    nu0 = torch.median(nu)

    time = torch.linspace(-t, t, x_size)
    n_training = int((N *70)/100)
    n_test = int((N*30)/100)

    dms_train = rand_interv(DM_min, DM_max, n_training)
    noises_train = rand_interv(3,8, n_training)
    swidth_train = rand_interv(4000,5000,n_training)
    width_train = 0.0000005*torch.ones(n_training)

    train_images = s.simulate(dms_train,width_train,nu,nu0,time,swidth_train, x_size, y_size, noises_train ,plot_flag=False)

    dms_test = rand_interv(DM_min, DM_max, n_test)
    noises_test = rand_interv(3,8, n_test)
    swidth_test = rand_interv(4000,5000,n_test)
    width_test = 0.0000005*torch.ones(n_test)

    test_images = s.simulate(dms_test,width_test,nu,nu0,time, swidth_test,x_size, y_size, noises_test, plot_flag=False)

    training_dataset = SimulatedDataset(train_images,dms_train,swidth_train)
    testing_dataset = SimulatedDataset(test_images,dms_test,swidth_test)

    if test_flag: 
        rand_index = torch.randint(0,50,(3,))
        super_test = []
        for val in rand_index:
            super_test.append(testing_dataset[val])
        for val in super_test: 
            print(' DM_obs {:.4f}.  SWidth_obs {:.4f} '.format(val[1],val[2]))
            print(torch.max(val[0]),torch.std(val[0]))
            plt.imshow(val[0].detach().numpy(),aspect='auto')
            plt.show()

    train_dataloader = DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
    testing_dataloader = DataLoader(testing_dataset,batch_size=batch_size,shuffle=True)

    return training_dataset, testing_dataset, train_dataloader,testing_dataloader, time, nu, nu0

def CreateDataset_noswidth(DM_min,DM_max, nu_i,nu_f, N, t, x_size, y_size, batch_size, noise = 7, test_flag=False):

    nu = torch.linspace(nu_i, nu_f, y_size)
    nu0 = torch.median(nu)

    time = torch.linspace(-t, t, x_size)
    n_training = int((N *70)/100)
    n_test = int((N*30)/100)

    dms_train = rand_interv(DM_min, DM_max, n_training)
    noises_train = rand_interv(3,8, n_training)
    swidth_train = rand_interv(4000,5000,n_training)
    width_train = 0.000001*torch.ones(n_training)

    train_images = s.simulate_noswidth(dms_train,width_train,nu,nu0,time,swidth_train, x_size, y_size, noises_train ,plot_flag=False)

    dms_test = rand_interv(DM_min, DM_max, n_test)
    noises_test = rand_interv(3,8, n_test)
    swidth_test = rand_interv(4000,5000,n_test)
    width_test = 0.000001*torch.ones(n_test)

    test_images = s.simulate_noswidth(dms_test,width_test,nu,nu0,time, swidth_test,x_size, y_size, noises_test, plot_flag=False)

    training_dataset = SimulatedDataset(train_images,dms_train,swidth_train)
    testing_dataset = SimulatedDataset(test_images,dms_test,swidth_test)

    if test_flag: 
        rand_index = torch.randint(0,50,(3,))
        super_test = []
        for val in rand_index:
            super_test.append(testing_dataset[val])
        for val in super_test: 
            print(' DM_obs {:.4f}.  SWidth_obs {:.4f} '.format(val[1],val[2]))
            print(torch.max(val[0]),torch.std(val[0]))
            plt.imshow(val[0].detach().numpy(),aspect='auto')
            plt.show()

    train_dataloader = DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
    testing_dataloader = DataLoader(testing_dataset,batch_size=batch_size,shuffle=True)

    return training_dataset, testing_dataset, train_dataloader,testing_dataloader, time, nu, nu0
    


if __name__=='__main__':
    print('Run dataset.py')