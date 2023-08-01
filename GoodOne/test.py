import dataset as dt
from model_def import train, VariationalAutoEncoder, save_model, VariationalAutoEncoder_noswidth
import argparse 
import torch 

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

train_dataset, test_dataset, train_dataloader, test_dataloader, t, nu, nu0 = dt.CreateDataset_noswidth(args.DM_min,args.DM_max, args.nu_i, args.nu_f, 
                                                    args.size, args.t, args.x_size, args.y_size, args.batch_size, test_flag=True )

print('Created the dataset')

load = True


input_dim = args.x_size * args.y_size                                   
model = VariationalAutoEncoder_noswidth(input_dim, args.x_size, args.y_size, nu, nu0,t)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


if load: 
    print('Loading the model')
    model.load_state_dict(torch.load('FRBAEGPU10010815:02.pt',map_location=torch.device(device)))

model.to(device)

print('Created the model')

lr = 1e-4

optimizer = torch.optim.Adam(model.parameters(),lr=lr)

model = train(model, args.epochs, train_dataloader, test_dataloader,optimizer)

print('Finished training')

save_model(model, args.batch_size)

