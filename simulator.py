import torch
from scipy.constants import e, pi, epsilon_0, parsec, m_e, c

import matplotlib.pyplot as plt

def gaussian(t, t0, sigma):
    """Compute the Gaussian function."""
    return torch.exp(-(t-t0).square() / sigma)

DM_CONST = 1e-6 * (e**2) / (8 * pi**2 * epsilon_0 * m_e * c) * parsec

def disp_delay(freq, dm, disp_ind=2.0):
    """Compute the dispersion delay (s) as a function of frequency (MHz) and DM."""

    return DM_CONST * dm / (freq**disp_ind)


def simulate(dm, width, nu, nu0, t, swidth, x_size, y_size,  noise=torch.tensor([0]), plot_flag=True):
    """
    Simulate FRBs based on input parameters.

    Args:
    dm (torch.tensor): Tensor containing DM values.
    width (torch.tensor): Tensor containing width values.
    nu (torch.tensor): Tensor containing nu values.
    nu0 (torch.tensor): Tensor containing the median of nu.
    t (torch.tensor): Tensor containing the time range.
    plot_flag (bool): Optional flag to control plotting. Default is True.

    Returns:
    frb (torch.tensor): Tensor containing the computed FRB values.
    """
    # Check input types
    assert isinstance(dm, torch.Tensor), "dm must be a torch.Tensor"
    assert isinstance(width, torch.Tensor), "width must be a torch.Tensor"
    assert isinstance(nu, torch.Tensor), "nu must be a torch.Tensor"
    assert isinstance(nu0, torch.Tensor), "nu0 must be a torch.Tensor"
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor"
    assert isinstance(noise,torch.Tensor),"noise must be a torch.Tensor"
    assert isinstance(plot_flag, bool), "plot_flag must be a boolean"
    
    # Check input shapes (one-dimensional tensors)
    assert dm.dim() == 1, "dm must be a one-dimensional tensor"
    assert width.dim() == 1, "width must be a one-dimensional tensor"
    assert nu.dim() == 1, "nu must be a one-dimensional tensor"
    assert nu0.dim() == 0, "nu0 must be a scalar tensor"
    assert t.dim() == 1, "t must be a one-dimensional tensor"
    
    # Calculate delay
    delay = disp_delay(nu0, dm[:, None]) - disp_delay(nu, dm[:, None])
    delay = delay.flip(1)
    

    noise_amp = noise[:,None] * torch.ones(1,x_size)
    normal_error = noise_amp[:,None]/200*torch.normal(mean=0, std=1, size=(len(noise), y_size,x_size))

    y = torch.linspace(0,y_size,y_size)

    y_cropping = 0.48*gaussian(y,y_size/2,swidth[:,None])
    y_cropping = y_cropping.view(-1,y_size,1)

    #Missing values
    number_zeros = torch.randint(y_size,(int(y_size/4),))
    zeros_index = torch.randint(y_size,(len(number_zeros),))
    zeros = torch.ones(y_size)
    zeros[zeros_index] = 0
    zeros = zeros.view(-1,y_size,1)

 
    # Compute FRB values
    frb = gaussian(t[None, None, :], -delay[:, :, None], width[:, None, None])
    
    frb = frb * y_cropping + normal_error
    frb = frb * zeros
    data_norm = frb.clone().detach()

    for i,image in enumerate(frb): 
        data_norm[i] = (image - torch.mean(image))/torch.std(image)
        #data_norm[i] = torch.nn.functional.normalize(image)

    # Plot images if the flag is True
    if plot_flag:
        for image in data_norm:
            plt.imshow(image, aspect='auto')
            plt.colorbar()
            plt.show()
            plt.close()
       
    
    return data_norm

def decoder(dm, width, nu, nu0, t, swidth,x_size, y_size,  noise=torch.tensor([0]),  plot_flag=True):
    """
    Simulate FRBs based on input parameters.

    Args:
    dm (torch.tensor): Tensor containing DM values.
    width (torch.tensor): Tensor containing width values.
    nu (torch.tensor): Tensor containing nu values.
    nu0 (torch.tensor): Tensor containing the median of nu.
    t (torch.tensor): Tensor containing the time range.
    plot_flag (bool): Optional flag to control plotting. Default is True.

    Returns:
    frb (torch.tensor): Tensor containing the computed FRB values.
    """
    # Check input types
    assert isinstance(dm, torch.Tensor), "dm must be a torch.Tensor"
    assert isinstance(width, torch.Tensor), "width must be a torch.Tensor"
    assert isinstance(nu, torch.Tensor), "nu must be a torch.Tensor"
    assert isinstance(nu0, torch.Tensor), "nu0 must be a torch.Tensor"
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor"
    assert isinstance(noise,torch.Tensor),"noise must be a torch.Tensor"
    assert isinstance(plot_flag, bool), "plot_flag must be a boolean"
    
    # Check input shapes (one-dimensional tensors)
    assert dm.dim() == 1, "dm must be a one-dimensional tensor"
    assert width.dim() == 1, "width must be a one-dimensional tensor"
    assert nu.dim() == 1, "nu must be a one-dimensional tensor"
    assert nu0.dim() == 0, "nu0 must be a scalar tensor"
    assert t.dim() == 1, "t must be a one-dimensional tensor"
    #print('DM:', dm)
    # Calculate delay
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nu = nu.to(device)
    t = t.to(device)

    delay = disp_delay(nu0, dm[:, None]) - disp_delay(nu, dm[:, None])
    delay = delay.flip(1)

    

    delay = delay.to(device)


    noise_amp = noise[:,None] * torch.ones(1,x_size)
    normal_error = noise_amp[:,None]/300*torch.normal(mean=0, std=1, size=(len(noise), y_size,x_size))
    normal_error = normal_error.to(device)

    y = torch.linspace(0,y_size,y_size)

    y = y.to(device)

    y_cropping = gaussian(y,y_size/2,swidth[:,None])
    y_cropping = torch.nn.functional.normalize(y_cropping)
    y_cropping = y_cropping.view(-1,y_size,1)

    y_cropping = y_cropping.to(device)
 
    # Compute FRB values
    frb = gaussian(t[None, None, :], -delay[:, :, None], width[:, None, None])
    
    frb = frb * y_cropping + normal_error

    data_norm = frb.clone().detach()

    for i,image in enumerate(frb): 
        data_norm[i] = (image - torch.mean(image))/torch.std(image)
        #data_norm[i] = torch.nn.functional.normalize(image) 

    # Plot images if the flag is True
    if plot_flag:
        for image in data_norm:
            plt.imshow(image, aspect='auto')
            plt.colorbar()
            plt.show()
            plt.close()
       
    
    return frb

def simulate_noswidth(dm, width, nu, nu0, t,swidth, x_size, y_size,  noise=torch.tensor([0]), plot_flag=True):
    """
    Simulate FRBs based on input parameters.

    Args:
    dm (torch.tensor): Tensor containing DM values.
    width (torch.tensor): Tensor containing width values.
    nu (torch.tensor): Tensor containing nu values.
    nu0 (torch.tensor): Tensor containing the median of nu.
    t (torch.tensor): Tensor containing the time range.
    plot_flag (bool): Optional flag to control plotting. Default is True.

    Returns:
    frb (torch.tensor): Tensor containing the computed FRB values.
    """
    # Check input types
    assert isinstance(dm, torch.Tensor), "dm must be a torch.Tensor"
    assert isinstance(width, torch.Tensor), "width must be a torch.Tensor"
    assert isinstance(nu, torch.Tensor), "nu must be a torch.Tensor"
    assert isinstance(nu0, torch.Tensor), "nu0 must be a torch.Tensor"
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor"
    assert isinstance(noise,torch.Tensor),"noise must be a torch.Tensor"
    assert isinstance(plot_flag, bool), "plot_flag must be a boolean"
    
    # Check input shapes (one-dimensional tensors)
    assert dm.dim() == 1, "dm must be a one-dimensional tensor"
    assert width.dim() == 1, "width must be a one-dimensional tensor"
    assert nu.dim() == 1, "nu must be a one-dimensional tensor"
    assert nu0.dim() == 0, "nu0 must be a scalar tensor"
    assert t.dim() == 1, "t must be a one-dimensional tensor"
    
    # Calculate delay
    delay = disp_delay(nu0, dm[:, None]) - disp_delay(nu, dm[:, None])
    delay = delay.flip(1)
    

    noise_amp = noise[:,None] * torch.ones(1,x_size)
    normal_error = noise_amp[:,None]*torch.normal(mean=1, std=1, size=(len(noise), y_size,x_size))

    y = torch.linspace(0,y_size,y_size)

    y_cropping = 0.48*gaussian(y,y_size/2,swidth[:,None])
    y_cropping = y_cropping.view(-1,y_size,1)

    #Missing values
    number_zeros = torch.randint(y_size,(int(y_size/4),))
    zeros_index = torch.randint(y_size,(len(number_zeros),))
    zeros = torch.ones(y_size)
    zeros[zeros_index] = 0
    zeros = zeros.view(-1,y_size,1)

 
    # Compute FRB values
    frb = gaussian(t[None, None, :], -delay[:, :, None], width[:, None, None])
    
    #frb = frb
    frb = frb * zeros
    data_norm = frb.clone().detach()
    snr = 1

    for i,image in enumerate(frb): 
        #data_norm[i] = (image - torch.mean(image))/torch.std(image)
        data_norm[i] = torch.nn.functional.normalize(image)
        current_snr = torch.mean(data_norm[i])/torch.std(normal_error[i])
        normal_error[i] = normal_error[i]*current_snr/snr
        data_norm[i] = torch.nn.functional.normalize(data_norm[i]+10*normal_error[i])


    # Plot images if the flag is True
    if plot_flag:
        for image in data_norm:
            plt.imshow(image, aspect='auto')
            plt.colorbar()
            plt.show()
            plt.close()
       
    
    return data_norm

def decoder_noswidth(dm, width, nu, nu0, t, swidth,x_size, y_size,  noise=torch.tensor([0]),  plot_flag=True):
    """
    Simulate FRBs based on input parameters.

    Args:
    dm (torch.tensor): Tensor containing DM values.
    width (torch.tensor): Tensor containing width values.
    nu (torch.tensor): Tensor containing nu values.
    nu0 (torch.tensor): Tensor containing the median of nu.
    t (torch.tensor): Tensor containing the time range.
    plot_flag (bool): Optional flag to control plotting. Default is True.

    Returns:
    frb (torch.tensor): Tensor containing the computed FRB values.
    """
    # Check input types
    assert isinstance(dm, torch.Tensor), "dm must be a torch.Tensor"
    assert isinstance(width, torch.Tensor), "width must be a torch.Tensor"
    assert isinstance(nu, torch.Tensor), "nu must be a torch.Tensor"
    assert isinstance(nu0, torch.Tensor), "nu0 must be a torch.Tensor"
    assert isinstance(t, torch.Tensor), "t must be a torch.Tensor"
    assert isinstance(noise,torch.Tensor),"noise must be a torch.Tensor"
    assert isinstance(plot_flag, bool), "plot_flag must be a boolean"
    
    # Check input shapes (one-dimensional tensors)
    assert dm.dim() == 1, "dm must be a one-dimensional tensor"
    assert width.dim() == 1, "width must be a one-dimensional tensor"
    assert nu.dim() == 1, "nu must be a one-dimensional tensor"
    assert nu0.dim() == 0, "nu0 must be a scalar tensor"
    assert t.dim() == 1, "t must be a one-dimensional tensor"
    #print('DM:', dm)
    # Calculate delay
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nu = nu.to(device)
    t = t.to(device)

    delay = disp_delay(nu0, dm[:, None]) - disp_delay(nu, dm[:, None])
    delay = delay.flip(1)

    delay = delay.to(device)

    noise_amp = noise[:,None] * torch.ones(1,x_size)
    normal_error = noise_amp[:,None]/300*torch.normal(mean=0, std=1, size=(len(noise), y_size,x_size))
    normal_error = normal_error.to(device)

    y = torch.linspace(0,y_size,y_size)

    y = y.to(device)

    y_cropping = gaussian(y,y_size/2,swidth[:,None])
    y_cropping = torch.nn.functional.normalize(y_cropping)
    y_cropping = y_cropping.view(-1,y_size,1)

    y_cropping = y_cropping.to(device)
 
    # Compute FRB values
    frb = gaussian(t[None, None, :], -delay[:, :, None], width[:, None, None])
    
    frb = frb  

    data_norm = frb.clone().detach()

    #for i,image in enumerate(frb): 
        #data_norm[i] = (image - torch.mean(image))/torch.std(image)
        #data_norm[i] = torch.nn.functional.normalize(image) 

    # Plot images if the flag is True
    if plot_flag:
        for image in data_norm:
            plt.imshow(image, aspect='auto')
            plt.colorbar()
            plt.show()
            plt.close()
       
    
    return data_norm



if __name__ == '__main__':
    print('Simulator implemented')