import dataset as dt
import argparse 

import matplotlib.pyplot as plt
import numpy as np
import torch 
import os


from model_def import VariationalAutoEncoder_noswidth, train, save_model_real_noise

#print(dt_string)


parser = argparse.ArgumentParser()

parser.add_argument('x_size', type=int, help='x_size of the images to be generated')
parser.add_argument('y_size', type=int , help='y_size of the images to be generated')
parser.add_argument('nu_i', type=float, help='Lower value of the frequency channels')
parser.add_argument('nu_f', type=float, help='Higher frequency channel')
parser.add_argument('t', type=float, help='Time from the central value of time')
parser.add_argument('size', type=float, help='Size of the dataset')
parser.add_argument('DM_min', type=float, help='Minimum of the DM')
parser.add_argument('DM_max', type=float, help='maximum of the DM')
parser.add_argument('batch_size', type = int, help='Indicate the batch size')
parser.add_argument('epochs', type = int, help='Indicate the number of epochs')

args = parser.parse_args()



if os.path.exists('/home/rvaldata/big_noise_test_images.pt'):
    print('Run on Snellius')
    test_dataloader = dt.LoadDataset('/home/rvaldata/big_noise_test_images.pt')
    train_dataloader = dt.LoadDataset('/home/rvaldata/big_noise_train_images.pt')
else:
    print('Run Locally')
    test_dataloader = dt.LoadDataset('/Users/ruggero/Desktop/GitModels/CombinedModels/2000_test_images.pt')
    train_dataloader = dt.LoadDataset('/Users/ruggero/Desktop/GitModels/CombinedModels/2000_train_images.pt')
    test_dataset = dt.LoadDataset('/Users/ruggero/Desktop/GitModels/CombinedModels/2000_test_dataset.pt')
    train_dataset = dt.LoadDataset('/Users/ruggero/Desktop/GitModels/CombinedModels/2000_train_dataset.pt') 

print('Loaded the dataset')

""" train_dataset, test_dataset, train_dataloader, test_dataloader, t, nu, nu0 = dt.CreateDataset_noswidth(args.DM_min,args.DM_max, args.nu_i, args.nu_f, 
                                                    args.size, args.t, args.x_size, args.y_size, args.batch_size, test_flag=True ) """

nu = torch.linspace(args.nu_i, args.nu_f, args.y_size)
nu0 = torch.median(nu)

time = torch.linspace(-0.1, 0.2, args.x_size)

input_dim = args.x_size * args.y_size                                  
model = VariationalAutoEncoder_noswidth(input_dim, args.x_size, args.y_size, nu, nu0,time)

load = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if load: 
    print('Loading the model')
    model.load_state_dict(torch.load('REALNOISE10180620:15.pt',map_location=torch.device(device)))

model.to(device)

print('Created the model')

lr = 1e-2

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

model = train(model, args.epochs, train_dataloader, test_dataloader, optimizer)

print('Finished training')

save_model_real_noise(model, args.batch_size)
