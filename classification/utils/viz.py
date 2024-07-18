import math
import matplotlib.pyplot as plt
import utils.utils as utils
import dynconv

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
unnormalize = utils.UnNormalize(mean, std)

def plot_image(input):
    ''' shows the first image of a 4D pytorch batch '''
    assert input.dim() == 4
    plt.figure('Image')
    im = unnormalize(input[0]).cpu().numpy().transpose(1,2,0)
    plt.imshow(im)


def plot_ponder_cost(masks):
    ''' plots ponder cost
    argument masks is a list with masks as returned by the network '''
    assert isinstance(masks, list)
    plt.figure('Ponder Cost')
    ponder_cost = dynconv.ponder_cost_map(masks)
    plt.imshow(ponder_cost, vmin=0, vmax=len(masks))
    plt.colorbar()

def plot_masks(masks):
    ''' plots individual masks as subplots 
    argument masks is a list with masks as returned by the network '''
    nb_mask = len(masks)
    WIDTH = 4
    HEIGHT = math.ceil(nb_mask / 4)
    f, axarr = plt.subplots(HEIGHT, WIDTH)

    for i, mask in enumerate(masks):
        x = i % WIDTH
        y = i // WIDTH

        # m = mask['std'].hard[0].cpu().numpy().squeeze(0)

        m = (mask['std'].soft[0] >= 0).float().cpu().numpy().squeeze(0) # hard is wrong...

        print ( np.mean(m))
        assert m.ndim == 2
        axarr[y,x].imshow(m, vmin=0, vmax=1)
        axarr[y,x].axis('off')
    
    for j in range(i+1, WIDTH*HEIGHT):
        x = j % WIDTH
        y = j // WIDTH
        f.delaxes(axarr[y,x])


from scipy import stats
import numpy as np
from sklearn.metrics import r2_score
def plot_mask_soft(masks):
    ''' plots individual masks as subplots 
    argument masks is a list with masks as returned by the network '''
    nb_mask = len(masks)
    WIDTH = 4
    HEIGHT = math.ceil(nb_mask / 4)
    f, axarr = plt.subplots(HEIGHT, WIDTH)

    for i, mask in enumerate(masks):
        x = i % WIDTH
        y = i // WIDTH

        m = mask['std'].soft[0].cpu().numpy().squeeze(0)

        assert m.ndim == 2
        axarr[y,x].imshow(m) #, vmin=0, vmax=1)

        axarr[y,x].axis('off')
    
    for j in range(i+1, WIDTH*HEIGHT):
        x = j % WIDTH
        y = j // WIDTH
        f.delaxes(axarr[y,x])


def plot_mask_distributions(masks):
    ''' plots individual masks as subplots 
    argument masks is a list with masks as returned by the network '''
    nb_mask = len(masks)
    WIDTH = 4
    HEIGHT = math.ceil(nb_mask / 4)
    f, axarr = plt.subplots(HEIGHT, WIDTH)

    for i, mask in enumerate(masks):
        x = i % WIDTH
        y = i // WIDTH

        m = mask['std'].soft[0].cpu().numpy().squeeze(0)

        assert m.ndim == 2
        # axarr[y,x].imshow(m) #, vmin=0, vmax=1)

        values = m.flatten()
        # axarr[y,x].hist(values, bins=50, density=True, alpha=0.75)


        # Assuming you have a 1D array of data named 'data'
        data = values

                
        # Fit a probability distribution to the data
        best_fit_distribution, best_fit_params = None, None
        best_fit_name = ""
        best_fit_r2 = -np.inf

        # List of candidate distributions to try
        candidate_distributions = [
            stats.norm,        # Normal distribution
            stats.expon,       # Exponential distribution
            stats.gamma,       # Gamma distribution
            stats.lognorm,     # Log-normal distribution
            stats.beta,        # Beta distribution
            # Add more distributions if needed
        ]

        # Fitting the distribution and selecting the best fit
        for distribution in candidate_distributions:
            # Fit the distribution to the data
            params = distribution.fit(data)
            
            # Separate parts of the parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            
            # Calculate the R-squared value as a measure of fit quality
            r2 = r2_score(data, distribution.pdf(data, loc=loc, scale=scale, *arg))
            
            # Select the best fit distribution based on R-squared value
            if r2 > best_fit_r2:
                best_fit_r2 = r2
                best_fit_name = distribution.name
                best_fit_params = params
                best_fit_distribution = distribution

                # Plotting the data and the best fit distribution
        # axarr[y,x].hist(data, bins=50, density=True, alpha=0.75, label='Data')
        # print( data.max()) # range paizei apo -10 , 10 sxedon
        # print( data.min())
        # axarr[y,x].hist(data, bins=16, density=True, range=(-10, 10), alpha=0.75, label='Data')
        axarr[y,x].hist(data, bins=16, density=True, alpha=0.75, label='Data')
        axarr[y,x].axvline(x = 0, color = 'b')
        xx = np.linspace(min(data), max(data), 1000)
        axarr[y,x].plot(xx, best_fit_distribution.pdf(xx, *best_fit_params), 'r-', label=best_fit_name)
        # axarr[y,x].axvline(x=0, color='black', linestyle='-')



        axarr[y,x].axis('off')
    
    for j in range(i+1, WIDTH*HEIGHT):
        x = j % WIDTH
        y = j // WIDTH
        f.delaxes(axarr[y,x])


def showKey():
    ''' 
    shows a plot, closable by pressing a key 
    '''
    plt.draw()
    plt.pause(1)
    input("<Hit Enter To Close>")
    plt.clf()
    plt.cla()
    plt.close('all')