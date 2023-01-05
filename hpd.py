import numpy as np
from ipdb import set_trace
import plotly.express as px
import scipy.stats as stats
import matplotlib
from sklearn import metrics
matplotlib.use("webagg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
import matplotlib.pyplot as plt


def get_mse(preds, labels):
    return metrics.mean_squared_error(labels, preds)


def search(data, labels, model):

    # 1. Get percents to iterate over
    start, end, increment = 0.5, 0.95, 0.05
    percents = np.arange(start, end, increment)[::-1]
    
    # 2. Create prior vars
    prior_indices, prior_mse, prior_p = {}, None, None 
    res = {}

    preds = model.predict(data)
    mse_on_full_test_data = get_mse(preds, labels)

    for feature in data.columns:

        for p in percents:

            # run HDP and get indices of data in percent
            *_, indices = hpd_grid(data[feature], percent=p)

            # Calculate mse just for the chosen indices
            if indices.size == 0:
                continue

            sliced_data = data.iloc[indices]
            sliced_labels = labels.iloc[indices]
            preds_on_sliced_data = model.predict(sliced_data)
            mse = get_mse(preds_on_sliced_data, sliced_labels)
        
            if prior_mse is not None:

                ## Save if change is more than 10%
                new_change = 100 * (mse - prior_mse) / prior_mse
                if new_change > 10 and mse > mse_on_full_test_data:
                    print(mse)

                    print(f'(max, min) values of {feature} in full data:' 
                          f'({data[feature].max(), data[feature].min()})')

                    print(f'(max, min) values of {feature} in sliced data:'
                          f'({sliced_data[feature].max(),' 
                          f'sliced_data[feature].min()})')

                    res[f'{feature}-{p}-{prior_p}'] =\
                        (prior_indices - set(indices), mse - prior_mse)
            
            # Reset
            prior_indices = set(indices)
            prior_mse = mse
            prior_p = p
      
    return res


def hpd_grid(sample, alpha=0.05, roundto=2, percent=0.5, show_plot=False):

    """Calculate highest posterior density (HPD) of array for given alpha.
    The HPD is the minimum width Bayesian credible interval (BCI).
    The function works for multimodal distributions, 
    returning more than one mode

    Parameters
    ----------
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    percent: float
        Perecent of data in the highest density region
    show_plots: bool
        if true, will show intermediary plots
    Returns
    ----------
    hpd: list with the highest density interval
    x: array with grid points where the density was evaluated
    y: array with the density values
    modes: list listing the values of the modes
    Src: 
    https://github.com/aloctavodia/BAP/blob/master/
    first_edition/code/Chp1/hpd.py
    Note this was modified to find low-accuracy areas
    """

    # data points that create a density plot when histogramed
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds on search space

    l = np.min(sample)
    u = np.max(sample)

    # get x-axis values
    x = np.linspace(l, u, 2000)

    # get kernel density estimate
    density = stats.gaussian_kde(sample)
    y = density.evaluate(x)

    
    if show_plot: 

        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(sample[:500])
        ax2.scatter(x=x, y=y) 

        plt.show()

    # sort by size of y (density estimate), descending 
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)

    # get all x's where y is in the top 1-alpha percent
    # this is to bound the type 1 error
    xy_cum_sum = 0
    hdv = [] # x values
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break

    # determine horizontal line corresponding to percent 
    yy_zipped = zip(y, y/np.sum(y))
    yy = sorted(yy_zipped, key=lambda x: x[1], reverse=True)

    y_cum_sum = 0
    y_cutoff = 0
    for val in yy:
        y_cum_sum += val[1]
        if y_cum_sum >= percent:
            y_cutoff = val[0]
            break

    # get indices of sample in range 
    intersections = []
    for i, curr in enumerate(y):
        prior = y[i-1]
        if (prior < y_cutoff and curr >= y_cutoff) or \
           (prior >= y_cutoff and curr < y_cutoff):
            intersections.append(x[i])

    indices = []
    for i in range(0, len(intersections), 2):
        lower, upper = intersections[i], intersections[i+1]
        indices.append(
                [i for i,v in enumerate(sample) if v <= upper and v >= lower]
                )

    # setup for difference comparison
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))

    # if y_i - y_{i-1} > diff then save
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))

    # prepare to calcualte value with highest density
    ite = iter(hpd)
    hpd = list(zip(ite, ite)) # create sequential pairs
    modes = []

    # find x and y value whith highest density
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto)) 
         # store x-value where density is highest in range

    indices = [item for sublist in indices for item in sublist]

    return hpd, x, y, modes, y_cutoff, np.array(indices)


