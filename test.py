import dataset as dt
from model_def import train, VariationalAutoEncoder
import argparse 
import torch 

from datetime import datetime


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

args = parser.parse_args()

train_dataset, test_dataset, train_dataloader, test_dataloader, t, nu, nu0 = dt.CreateDataset(args.DM_min,args.DM_max, args.nu_i, args.nu_f, 
                                                    args.size, args.t, args.x_size, args.y_size, args.batch_size, test_flag=False)

print('Created the dataset')


input_dim = args.x_size * args.y_size                                   
model = VariationalAutoEncoder(input_dim, args.x_size, args.y_size,nu, nu0,t)

print('Created the model')

lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

model = train(model, 10, train_dataloader, test_dataloader,optimizer)

now = datetime.now()
dt_string = now.strftime("%d%m %H:%M:%S")
model_save_name = 'FRBAEGPU '+str(args.batch_size)+dt_string+'.pt'

torch.save(model.state_dict(),model_save_name)


