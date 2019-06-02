import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.offline as py
import plotly.graph_objs as go
import colorlover as cl

sns.set()

################################################################################
#                               PROGRESS TRCAKER
################################################################################

def update_progress(progress,message=""):
    # update_progress() : Displays or updates a console progress bar
    ## Accepts a float between 0 and 1. Any int will be converted to a float.
    ## A value under 0 represents a 'halt'.
    ## A value at 1 or bigger represents 100%
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rProgress : [{0}] {1}% {2} {3}".format( "="*block +
                                                        " "*(barLength-block),
                                                         round(progress*100,2),
                                                         status,message)
    sys.stdout.write(text)
    sys.stdout.flush()

################################################################################
#                          PLOTS FOR estimate
################################################################################


def visualisation(df,model):

    predictors_name = np.array(['ED', 'SOUTH', 'NONWH', 'HISP', 'FE', 'MARR', 'MARRFE', 'EX',
        'UNION', 'MANUF', 'CONSTR', 'MANAG', 'SALES', 'CLER',
        'SERV', 'PROF'])
    trace1 = []
    colors = cl.scales[str(len(df.columns))]["qual"]["Dark2"]

    for i,name in enumerate(df.columns):
        trace0 = go.Scatter(
        y = df[name].values,
        name = name,
        mode = 'markers',
        marker = dict(
            size = 10,
            color = colors[i]
        ),
        opacity=1
        )
        trace1.append(trace0)

    if len(df)>16:
        layout = dict(title = 'Result comparison for model    {}'.format(model.name),
                  yaxis = dict(zeroline = True),
                   xaxis = go.layout.XAxis(
                            tickmode = 'array',
                            tickvals = np.arange(0,17),
                            ticktext = np.insert(predictors_name,0,"noise"),
                           zeroline = False
                   )
                 )
    else:
        layout = dict(title = 'Result comparison',
                  yaxis = dict(zeroline = True),
                   xaxis = go.layout.XAxis(
                            tickmode = 'array',
                            tickvals = np.arange(0,16),
                            ticktext = predictors_name,
                           zeroline = False
                   )
                 )

    fig = dict(data=trace1, layout=layout)
    py.iplot(fig)


################################################################################
#                          PLOTS FOR METROPOLIS HASTINGS
################################################################################


def big_plot(a = 16.0,b = 8.0):
    plt.rcParams['figure.figsize'] = (a, b)

def reset_plot():
    plt.rcParams['figure.figsize'] = (8.0, 4.0)


def compare_samples_MH(sample1,sample2):
    fig, axs = plt.subplots(4, 4, figsize=(16, 16), sharey=True, sharex = True)
    axs[0,0].plot(sample1[:,1], alpha = 0.8)
    axs[0,1].plot(sample1[:,2], alpha = 0.8)
    axs[0,2].plot(sample1[:,3], alpha = 0.8)
    axs[0,3].plot(sample1[:,4], alpha = 0.8)
    axs[0,0].plot(sample2[:,1], alpha = 0.8)
    axs[0,1].plot(sample2[:,2], alpha = 0.8)
    axs[0,2].plot(sample2[:,3], alpha = 0.8)
    axs[0,3].plot(sample2[:,4], alpha = 0.8)
    axs[1,0].plot(sample1[:,5], alpha = 0.8)
    axs[1,1].plot(sample1[:,6], alpha = 0.8)
    axs[1,2].plot(sample1[:,7], alpha = 0.8)
    axs[1,3].plot(sample1[:,8], alpha = 0.8)
    axs[1,0].plot(sample2[:,5], alpha = 0.8)
    axs[1,1].plot(sample2[:,6], alpha = 0.8)
    axs[1,2].plot(sample2[:,7], alpha = 0.8)
    axs[1,3].plot(sample2[:,8], alpha = 0.8)
    axs[2,0].plot(sample1[:,9], alpha = 0.8)
    axs[2,1].plot(sample1[:,10], alpha = 0.8)
    axs[2,2].plot(sample1[:,11], alpha = 0.8)
    axs[2,3].plot(sample1[:,12], alpha = 0.8)
    axs[2,0].plot(sample2[:,9], alpha = 0.8)
    axs[2,1].plot(sample2[:,10], alpha = 0.8)
    axs[2,2].plot(sample2[:,11], alpha = 0.8)
    axs[2,3].plot(sample2[:,12], alpha = 0.8)
    axs[3,0].plot(sample1[:,13], alpha = 0.8)
    axs[3,1].plot(sample1[:,14], alpha = 0.8)
    axs[3,2].plot(sample1[:,15], alpha = 0.8)
    axs[3,3].plot(sample1[:,16], alpha = 0.8)
    axs[3,0].plot(sample2[:,13], alpha = 0.8)
    axs[3,1].plot(sample2[:,14], alpha = 0.8)
    axs[3,2].plot(sample2[:,15], alpha = 0.8)
    axs[3,3].plot(sample2[:,16], alpha = 0.8)
    plt.ylim(-1.5,1.5)
    plt.tight_layout()
    plt.show()

def autocorrelation(time_series, maxRange):
    # estimate the autocorrelation

    l = len(time_series)
    ans = np.zeros(2*maxRange+1)
    delta = np.arange(-maxRange,maxRange+1,1)


    for k in range(2*maxRange+1):
        v0 = time_series[maxRange            : l - maxRange           ]
        v1 = time_series[maxRange - delta[k] : l - maxRange - delta[k]]

        m0 = np.mean(v0)
        m1 = np.mean(v1)
        cov = np.sum( (v0-m0) * (v1-m1) / len(v0) )
        var0 = np.sum( (v0-m0)**2 / len(v0) )
        var1 = np.sum( (v1-m1)**2 / len(v0) )
        corr = cov / (var0 * var1)**0.5

        ans[k] = corr

    return delta, ans

def showAutocorrelation(samples, delta = None, col = None):

    if delta == None:
        delta = np.int( len(samples) / 6 )
    _, trueCorrelation = autocorrelation(samples, delta )

    if col == None:
        plt.plot(np.arange(-delta,delta+1), trueCorrelation)
    else:
        plt.plot(np.arange(-delta,delta+1), trueCorrelation, c = col)
    plt.ylim([-1.1,1.1])


def samples_exploration(samples, iterations = True, distribution = True,
                    correlation = True, size_samples = (24,13), names = None):


    "visualisation of the metropolis algorithm"
    if len(samples.shape)>1:
        size = samples.shape[1]
    else:
        size = 1
    colors = sns.color_palette("hls", size)




    if iterations:
        print("iterations")
        rows = int(size/6)+1
        if rows > 30: big_plot(24,12)
        else: big_plot(24,6)
        for k in range(size):
            if names == None:
                plt.plot(samples[:,k], c = colors[k],label = "coord {}".format(k+1), alpha = 0.5)
            else :
                plt.plot(samples[:,k], c = colors[k], alpha = 0.5, label = names[k])

        plt.tight_layout()
        plt.legend()
        plt.show()
        reset_plot()

    if distribution:
        print("estimation of the distributions")
        rows = int(size/6)+1
        if rows > 30: big_plot(24,12)
        else: big_plot(24,6)

        for k in range(size):
            plt.subplot(rows,6,k+1)
            sns.distplot(samples[:,k], color = colors[k])
        plt.tight_layout()
        plt.show()

    if correlation:
        print("autocorrelation")
        big_plot(26,8)
        for k in range(size):
            showAutocorrelation(samples[:,k], col = colors[k])
        plt.show()
    reset_plot()
